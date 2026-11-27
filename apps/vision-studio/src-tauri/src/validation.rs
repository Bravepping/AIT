use super::{calibration, control::{now_ms, FRAME_TIMEOUT_MS}, diagnostics::StreamDiagnostics, hidden_process,
    AutomationController, Profile, Worker, MODEL_NAME};
use serde_json::{json, Value};
use std::{
    env, fs,
    io::{BufRead, BufReader, Write},
    path::Path,
    process::Stdio,
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

mod mock;
mod soak;

pub fn verify_control_stream() -> Result<(), String> { soak::verify() }

pub fn verify_transport_stream(report: u64, report_at_ms: u64) -> Result<(), String> {
    mock::write_transport(report, report_at_ms)
}

pub fn verify_live(source_name: &str, output: &Path, pause_marker: Option<&Path>) -> Result<(), String> {
    let mut report = verify_controller()?;
    let executable = env::current_exe().map_err(|error| error.to_string())?;
    let root = executable.parent().ok_or("缺少程序目录")?;
    let runtime = root.join("vision-runtime");
    let mut child = hidden_process(&runtime.join("python.exe"))
        .args(["-B", "-X", "utf8", "-u"])
        .arg(runtime.join("worker.py"))
        .arg("--desktop")
        .current_dir(root)
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| error.to_string())?;
    let stdout = child.stdout.take().ok_or("缺少推理输出")?;
    let stdin = child.stdin.take().ok_or("缺少推理输入")?;
    let mut worker = Worker { child, stdin, model_path: root.join(MODEL_NAME) };
    let (sender, receiver) = mpsc::sync_channel(32);
    thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if let Ok(event) = serde_json::from_str::<Value>(&line) {
                if sender.send(event).is_err() {
                    break;
                }
            }
        }
    });
    let mut send = |message: Value| {
        writeln!(worker.stdin, "{message}").map_err(|error| error.to_string())
    };
    let wait = |kind: &str, seconds| -> Result<Value, String> {
        let deadline = Instant::now() + Duration::from_secs(seconds);
        loop {
            if Instant::now() >= deadline {
                return Err(format!("等待 {kind} 超时"));
            }
            let event = receiver.recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .map_err(|error| format!("等待 {kind} 失败，{error}"))?;
            if event["kind"] == kind {
                if let Some(error) = event["error"].as_str() {
                    return Err(error.into());
                }
                if kind != "detections" || (event["boxes"].as_array().is_some_and(|boxes| !boxes.is_empty())
                    && event["frame_at"].as_u64().is_some_and(crate::control::frame_fresh))
                {
                    return Ok(event);
                }
            }
        }
    };
    wait("ready", 10)?;
    send(json!({"cmd":"discover"}))?;
    let sources = wait("sources", 15)?;
    let selected = sources["sources"].as_array()
        .and_then(|items| items.iter().find(|item| item[0] == source_name))
        .ok_or("找不到指定的真实 NDI 源")?;
    send(json!({"cmd":"model","model":1,"path":root.join(MODEL_NAME),"provider":"TensorRT"}))?;
    let model = wait("model", 360)?;
    if model["provider"] != "TensorRT" {
        return Err("实际后端不是 TensorRT".into());
    }
    send(json!({"cmd":"stream","source":1,"name":selected[0],"url":selected[1]}))?;
    let stream = wait("stream_state", 15)?;
    if stream["state"] != "live" {
        return Err("真实 NDI 源未进入接收状态".into());
    }
    let frame = wait("detections", 90)?;
    let count = frame["boxes"].as_array().map(Vec::len);
    let mut gate = AutomationController::default();
    gate.request_model(1);
    gate.request_stream(1);
    gate.worker(&model);
    gate.worker(&stream);
    gate.worker(&json!({"kind":"ndi_health","source":1,"frame_at":frame["frame_at"]}));
    gate.worker(&frame);
    let profile: Profile = serde_json::from_value(json!({
        "mask":3,"toggle":false,"enabled_classes":[0,1,2,3,4,5,6],"classes":null,
        "priority":"closest","gain_x":1.0,"gain_y":1.0,"max_step":24,
        "dead_zone_px":2.0,"smooth_half_life_ms":12.0
    })).map_err(|error| error.to_string())?;
    gate.configure(vec![profile])?;
    let mut buttons = input(0, 0);
    gate.buttons(&buttons);
    gate.arm()?;
    buttons["buttons"] = json!(3);
    let armed = gate.buttons(&buttons).ok_or("现场联调未触发组合键")?;
    if let Some(marker) = pause_marker {
        check(armed["state"] == "Running" && !armed["candidate"].is_null(), "暂停验证前没有候选位移")?;
        return verify_paused_ndi(&receiver, &mut gate, &buttons, marker, output);
    }
    let deadline = Instant::now() + Duration::from_secs(6);
    let mut latencies = Vec::new();
    let mut decisions = Vec::new();
    let mut positions = std::collections::BTreeSet::new();
    let mut locked = 0;
    let mut first = None;
    while Instant::now() < deadline {
        let Ok(event) = receiver.recv_timeout(deadline.saturating_duration_since(Instant::now())) else { break };
        if event["kind"] == "detections" {
            gate.buttons(&buttons);
        }
        if let Some(result) = gate.worker(&event) {
            let target = &result["candidate"];
            if let Some(ms) = target["latency_ms"].as_u64() {
                latencies.push(ms);
                decisions.push(target["decision_ms"].as_f64().unwrap());
                positions.insert(target["anchor"].to_string());
            locked += usize::from(target["selection_reason"].as_str().is_some_and(|reason| reason.starts_with("延续")));
                first.get_or_insert_with(|| target.clone());
            }
        }
        gate.tick();
    }
    if latencies.is_empty() {
        return Err("未能采集候选位移延迟".into());
    }
    let before_release = wait("detections", 5)?;
    gate.worker(&json!({"kind":"ndi_health","source":1,"frame_at":before_release["frame_at"]}));
    gate.buttons(&buttons);
    gate.worker(&before_release);
    check(gate.target_sample().is_some(), "松键测试前没有候选")?;
    let release = gate.buttons(&input(2, 0)).ok_or("模拟松键没有返回状态")?;
    check(release["candidate"].is_null() && release["active"] == false, "松开左键后候选仍有效")?;
    let after_release = wait("detections", 5)?;
    gate.worker(&json!({"kind":"ndi_health","source":1,"frame_at":after_release["frame_at"]}));
    gate.worker(&after_release);
    check(gate.target_sample().is_none(), "松键后的新检测帧重新产生了候选")?;
    latencies.sort_unstable();
    let samples = latencies.len();
    report["receiver_to_candidate_ms"] = json!({
        "samples": samples,
        "average": latencies.iter().sum::<u64>() as f64 / samples as f64,
        "p95": latencies[(samples * 95).div_ceil(100) - 1],
        "p99": latencies[(samples * 99).div_ceil(100) - 1],
    });
    report["detected_boxes"] = json!(count);
    report["frame_origin"] = json!("real live NDI");
    report["live_hotkey_input"] = json!("simulated physical mouse reports");
    report["live_candidate_cleared_on_single_key_release"] = json!(true);
    report["distinct_target_positions"] = json!(positions.len());
    report["lock_continuations"] = json!(locked);
    report["example_candidate"] = json!(first);
    report["decision_mean_ms"] = json!(decisions.iter().sum::<f64>() / decisions.len() as f64);
    report["dry_run_scheduled"] = json!(gate.movement.submitted);
    report["provider"] = model["provider"].clone();
    report["backend"] = model["backend"].clone();
    report["source"] = json!(source_name);
    report["preview_fps"] = json!(0);
    fs::write(output, serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?)
        .map_err(|error| error.to_string())
}

fn verify_paused_ndi(receiver: &mpsc::Receiver<Value>, gate: &mut AutomationController,
    buttons: &Value, marker: &Path, output: &Path) -> Result<(), String>
{
    fs::write(marker, "DRY_RUN_RUNNING").map_err(|error| error.to_string())?;
    let deadline = Instant::now() + Duration::from_secs(12);
    let mut last_capture = None;
    let mut watchdog = None;
    let mut sdk_gap = None;
    let mut resumed = false;
    while Instant::now() < deadline {
        match receiver.recv_timeout(Duration::from_millis(10)) {
            Ok(event) => {
                if event["kind"] == "ndi_health" {
                    last_capture = event["frame_at"].as_u64();
                } else if event["kind"] == "stream_state" && event["state"] == "lost" {
                    check(watchdog.is_some(), "安全锁止没有早于 SDK 断流通知")?;
                    check(gate.worker(&event).is_some_and(|state| state["candidate"].is_null()), "断流后仍有候选")?;
                    sdk_gap = event["gap_ms"].as_u64();
                    fs::write(marker, "NDI_LOST").map_err(|error| error.to_string())?;
                    continue;
                } else if event["kind"] == "stream_state" && event["state"] == "resumed" {
                    resumed = sdk_gap.is_some();
                }
                let detections = event["kind"] == "detections";
                if detections {
                    gate.buttons(buttons);
                }
                gate.worker(&event);
                if resumed && detections && event["boxes"].as_array().is_some_and(|boxes| !boxes.is_empty()) {
                    gate.tick();
                    check(gate.target_sample().is_none(), "NDI 恢复后自动恢复了跟随")?;
                    check(gate.arm()?["candidate"].is_null(), "重新武装时沿用了已经按下的组合")?;
                    gate.buttons(&input(0, 0));
                    gate.buttons(buttons);
                    check(gate.target_sample().is_some(), "重新武装和重新按键后未恢复跟随")?;
                    let (lock_age, reason) = watchdog.ok_or("缺少安全锁止记录")?;
                    let result = json!({"status":"PASS","source":"simulated local NDI sender pause and resume",
                        "buttons":"simulated","watchdog_threshold_ms":FRAME_TIMEOUT_MS,
                        "observed_frame_age_at_lock_ms":lock_age,"lock_reason":reason,"sdk_gap_ms":sdk_gap,
                        "queued_candidate_cleared":true,"rearm_required":true,"resumed_after_manual_rearm":true});
                    return fs::write(output, serde_json::to_vec_pretty(&result).map_err(|error| error.to_string())?)
                        .map_err(|error| error.to_string());
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(error) => return Err(error.to_string()),
        }
        if let Some(state) = gate.tick() {
            if state["state"] == "Fault" {
                check(matches!(state["reason"].as_str(), Some("NDI 最新帧超时" | "推理结果已过期"))
                    && state["candidate"].is_null(),
                    "安全锁止的原因或队列清空结果异常")?;
                watchdog = last_capture.and_then(|timestamp| now_ms().checked_sub(timestamp))
                    .map(|age| (age, state["reason"].clone()));
            }
        }
    }
    Err("未观测到完整的 NDI 停帧与安全锁止序列".into())
}

fn fixture_profile(id: u32, mask: u8, toggle: bool, classes: Value) -> Profile {
    serde_json::from_value(json!({
        "id":id,"name":format!("模拟方案 {id}"), "mask":mask, "toggle":toggle,
        "enabled_classes":[1,2], "classes":classes, "priority":"confidence",
        "confidence":0.65,"anchor_x":25.0,"anchor_y":-25.0,"range_percent":50.0,
        "gain_x":2.0,"gain_y":0.5,"invert_x":true,"max_step":24,"switch_confirm_frames":1
    })).unwrap()
}

fn input(buttons: u8, age_ms: u64) -> Value {
    json!({"buttons":buttons,"connected":true,"monitoring":true,"halted":false,"age_ms":age_ms})
}

fn fixture_frame(sequence: u64, boxes: &[[f64; 6]]) -> Value {
    json!({"kind":"detections","width":640,"height":640,"source":1,"model":1,
        "sequence":sequence,"frame_at":now_ms(),"boxes":boxes})
}

fn verify_controller() -> Result<Value, String> {
    let mut gate = AutomationController::default();
    gate.request_model(1);
    gate.request_stream(1);
    gate.worker(&json!({"kind":"model","model":1,"provider":"TensorRT"}));
    gate.worker(&json!({"kind":"stream_state","state":"live","source":1,"frame_at":now_ms()}));
    let profiles = || vec![
        fixture_profile(1, 3, false, Value::Null), fixture_profile(2, 8, true, json!([2])),
        fixture_profile(3, 1, false, json!([1, 2])),
    ];
    gate.configure(profiles())?;
    let conflict = gate.configure(vec![fixture_profile(1, 8, true, json!([1])),
        fixture_profile(2, 8, false, json!([2]))]);
    check(conflict.is_err(), "重复热键未被拒绝")?;
    let center = [[300., 300., 340., 340., 0.92, 1.], [360., 300., 390., 330., 0.83, 2.],
                  [310., 310., 330., 330., 0.64, 1.], [10., 10., 35., 35., 0.99, 1.]];
    gate.worker(&fixture_frame(1, &center));
    let ready = gate.buttons(&input(0, 0)).ok_or("未报告就绪")?;
    check(ready["state"] == "Ready" && ready["real_control_available"] == false, "就绪状态异常")?;
    let armed = gate.arm()?;
    check(armed["state"] == "Armed", "启用状态异常")?;
    gate.buttons(&input(2, 0));
    let started = gate.buttons(&input(3, 0)).ok_or("组合键没有开始")?;
    check(started["state"] == "Running" && started["profile_id"] == 1, "多方案优先级失效")?;
    let original = started["candidate"].clone();
    check(original["class"] == 1 && original["relative"] == json!([-20,-5]), "置信度、范围或坐标换算错误")?;
    let released = gate.buttons(&input(1, 0)).ok_or("松开没有输出")?;
    check(released["state"] == "Armed" && released["candidate"].is_null(), "松开右键后仍保留候选")?;
    gate.buttons(&input(0, 0));
    gate.buttons(&input(1, 0));
    let left_first = gate.buttons(&input(3, 0)).ok_or("先左后右没有开始")?;
    check(left_first["profile_id"] == 1, "组合键左键先按失败")?;
    let released = gate.buttons(&input(2, 0)).ok_or("松开左键没有输出")?;
    check(released["candidate"].is_null(), "松开左键后仍保留候选")?;
    gate.buttons(&input(0, 0));
    let toggle = gate.buttons(&input(8, 0)).ok_or("单键切换没有开始")?;
    check(toggle["profile_id"] == 2 && toggle["candidate"]["class"] == 2, "单类别切换失效")?;
    check(gate.buttons(&input(0,0)).is_none(), "Toggle 在松开时停止")?;
    let stopped = gate.buttons(&input(8, 0)).ok_or("第二次按下未停止")?;
    check(stopped["candidate"].is_null(), "Toggle 关闭后仍保留候选")?;
    gate.buttons(&input(0, 0));
    let tracking_started = gate.buttons(&input(3, 0)).ok_or("组合键重新按下没有启动跟踪")?;
    thread::sleep(Duration::from_millis(35));
    let moved = [[302., 301., 342., 341., 0.75, 1.], [360., 300., 390., 330., 0.99, 2.]];
    let follow = gate.worker(&fixture_frame(2, &moved)).ok_or("移动目标没有输出")?;
    check(follow["candidate"]["class"] == 1 && follow["candidate"]["anchor"] == json!([332.0,311.0])
        && follow["candidate"]["relative"] == json!([-24,-5])
        && follow["candidate"]["track_id"] == tracking_started["candidate"]["track_id"]
        && follow["candidate"]["selection_reason"] == "延续同一 Track ID", "移动时丢失锁定或选了旁边目标")?;
    let vanished = gate.worker(&fixture_frame(3, &moved[1..])).ok_or("目标消失没有立即清除")?;
    check(vanished["candidate"].is_null(), "失去锁定后保留了过期位移")?;
    let acquired = gate.worker(&fixture_frame(4, &moved[1..])).ok_or("下一帧没有重新选择目标")?;
    check(acquired["candidate"]["class"] == 2, "下一帧无法重新锁定")?;
    let mut delayed = input(3, 10);
    delayed["report_at_ms"] = json!(now_ms() - 1201);
    let stale = gate.buttons(&delayed).ok_or("过期按键没有触发故障")?;
    check(stale["state"] == "Fault" && stale["candidate"].is_null() && gate.arm().is_err(), "过期报文仍能触发")?;
    gate.buttons(&input(0, 0));
    gate.arm()?;
    gate.buttons(&input(3, 0));
    gate.worker(&json!({"kind":"stream_state","state":"live","source":1,"frame_at":now_ms() - FRAME_TIMEOUT_MS - 1}));
    let watchdog = gate.tick().ok_or("视频新鲜度保护没有触发")?;
    check(watchdog["reason"] == "NDI 最新帧超时" && watchdog["state"] == "Fault"
        && watchdog["candidate"].is_null(), "等待 SDK 断流事件或仍保留候选")?;
    gate.worker(&json!({"kind":"ndi_health","source":1,"frame_at":now_ms()}));
    gate.worker(&fixture_frame(5, &moved));
    check(gate.buttons(&input(3, 0)).is_none(), "新帧意外解除故障")?;
    gate.buttons(&input(0, 0));
    gate.arm()?;
    gate.buttons(&input(3, 0));
    let lost = gate.worker(&json!({"kind":"stream_state","state":"lost","source":1})).ok_or("断流没有触发故障")?;
    check(lost["state"] == "Fault" && lost["candidate"].is_null() && gate.arm().is_err(), "断流后仍能触发")?;
    let fault = gate.worker(&json!({"kind":"model","model":1,"error":"simulated"})).ok_or("模型错误没有触发故障")?;
    check(fault["state"] == "Fault" && fault["candidate"].is_null() && gate.arm().is_err(), "模型失败后仍能触发")?;
    let emergency = gate.disarm("simulated emergency stop");
    check(emergency["state"] == "Stopped" && emergency["candidate"].is_null(), "紧急停止未清除队列")?;
    let calibration = serde_json::to_value(calibration::estimate(-20.0, 5.0)?).map_err(|error| error.to_string())?;
    check(calibration == json!({"gain_x":0.5,"gain_y":2.0,"invert_x":true,"invert_y":false})
        && calibration::estimate(0.0, 5.0).is_err(), "校准比例或方向错误")?;
    let mut diagnostic = StreamDiagnostics::default();
    diagnostic.observe(&json!({"kind":"ndi_health","source":1,"frame_at":now_ms()-40,
        "raw_pending":1,"prepared_pending":0}), 1, None);
    let rate = |fps| json!({"kind":"stream","source":1,"nominal":240.0,"fps":fps});
    let dip = diagnostic.observe(&rate(200.0), 1, Some((now_ms() - 10, 44))).ok_or("降帧未记录")?;
    let recovery = diagnostic.observe(&rate(239.0), 1, None).ok_or("帧率恢复未记录")?;
    check(dip["duration_ms"].is_null() && dip["raw_pending"] == 1 && dip["prepared_pending"] == 0
        && dip["target_sequence"] == 44 && dip["frame_age_ms"].is_number() && recovery["duration_ms"].is_number()
        && recovery["target_sequence"].is_null(), "降帧诊断缺少状态或重复使用旧目标")?;
    Ok(json!({"status":"PASS","fixture":"simulated detections and button reports",
        "both_press_orders":true,"release_clears":true,"toggle":true,"multi_profile":true,
        "conflicts":true,"confidence":true,"selection_reason":true,"calibration":true,
        "range":true,"dynamic_lock":true,"vanish_clears":true,"reacquisition":true,
        "stale_input_fault":true,"ndi_watchdog_ms":FRAME_TIMEOUT_MS,"rearm_after_video_fault":true,
        "ndi_loss_fault":true,"model_fault":true,"emergency_stop":true,"rate_dip_fields":true,
        "mock_kmbox":mock::verify()?, "real_control_available":false}))
}

fn check(verified: bool, failure: &str) -> Result<(), String> {
    if verified { Ok(()) } else { Err(failure.into()) }
}

#[test]
fn controller_replay_regression() { verify_controller().unwrap(); }

pub fn verify_trace(input: &Path, output: &Path) -> Result<(), String> {
    let mut report = verify_controller()?;
    let bytes = fs::read(input).map_err(|error| error.to_string())?;
    let mut frame: Value = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    let mut controller = AutomationController::default();
    controller.configure(vec![serde_json::from_value(json!({
        "id":1,"mask":3,"enabled_classes":[0,1,2,3,4,5,6],
    })).map_err(|error| error.to_string())?])?;
    controller.request_model(1);
    controller.request_stream(1);
    controller.worker(&json!({"kind":"model","model":1,"provider":"TensorRT"}));
    controller.worker(&json!({"kind":"stream_state","state":"live","source":1,"frame_at":now_ms()}));
    frame["frame_at"] = json!(now_ms());
    controller.worker(&frame);
    controller.buttons(&self::input(0, 0));
    controller.arm()?;
    let outcome = controller.buttons(&self::input(3, 0)).ok_or("回放时未触发")?;
    check(!outcome["candidate"].is_null(), "回放时未成功选中目标")?;
    report["frame_origin"] = json!("replayed detection fixture");
    report["example_candidate"] = outcome["candidate"].clone();
    fs::write(output, serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?)
        .map_err(|error| error.to_string())
}
