use crate::movement::{KmBoxDriver, KmBoxPipe, MoveContext, RelativeMove, Smoothing};
use super::{check, fixture_frame, input, now_ms, AutomationController, Duration, Instant};
use serde_json::{json, Value};

#[derive(Default)]
struct MockKmBox {
    sent: Vec<RelativeMove>,
    disconnected: bool,
    live: bool,
}

impl KmBoxDriver for MockKmBox {
    fn submit_relative(&mut self, command: RelativeMove, _: MoveContext) -> Result<(), &'static str> {
        if self.disconnected { return Err("Mock KMBox 发送失败"); }
        self.sent.push(command);
        Ok(())
    }

    fn start_live(&mut self) -> Result<(), &'static str> {
        if self.disconnected { return Err("Mock KMBox 未连接"); }
        self.live = true;
        Ok(())
    }

    fn stop_live(&mut self) { self.live = false; }
}

const FIRST: [[f64; 6]; 1] = [[320., 320., 360., 360., 0.8, 1.]];
const NEXT: [[f64; 6]; 1] = [[325., 328., 365., 368., 0.9, 1.]];

fn send_frame<D: KmBoxDriver>(gate: &mut AutomationController<D>, sequence: u64, boxes: &[[f64; 6]]) -> Option<Value> {
    gate.worker(&json!({"kind":"ndi_health","source":1,"frame_at":now_ms()}));
    gate.worker(&fixture_frame(sequence, boxes))
}

fn setup<D: KmBoxDriver>(driver: D) -> Result<AutomationController<D>, String> {
    let mut gate = AutomationController::with_driver(driver);
    gate.request_model(1);
    gate.request_stream(1);
    gate.worker(&json!({"kind":"model","model":1,"provider":"TensorRT"}));
    gate.worker(&json!({"kind":"stream_state","state":"live","source":1,"frame_at":now_ms()}));
    let profile = json!({
        "name":"模拟设备", "mask":3, "enabled_classes":[1,2], "priority":"confidence", "max_step":24,
        "switch_confirm_frames":2,
    });
    gate.configure(vec![serde_json::from_value(profile).map_err(|error| error.to_string())?])?;
    send_frame(&mut gate, 1, &FIRST);
    gate.buttons(&input(0, 0));
    gate.arm()?;
    Ok(gate)
}

fn verify_tracking() -> Result<(), String> {
    let mut gate = setup(MockKmBox::default())?;
    let start = gate.buttons(&input(3, 0)).ok_or("缺少首次锁定")?;
    check(start["candidate"]["lock_id"] == 1, "初次锁定编号错误")?;
    let original = [328., 327., 368., 367., 0.75, 1.];
    let challenger = [195., 262., 228., 303., 0.95, 2.];
    let other = [410., 350., 450., 395., 0.99, 2.];
    send_frame(&mut gate, 2, &[original, challenger]);
    check(gate.target_sample().is_some_and(|sample| sample.0 == 2), "新目标抢占了有效锁定")?;
    let lost = send_frame(&mut gate, 3, &[challenger]).ok_or("缺少丢失更新")?;
    check(lost["candidate"].is_null(), "丢失后留有位移")?;
    send_frame(&mut gate, 4, &[challenger]);
    check(gate.target_sample().is_none(), "只确认一帧就切换")?;
    let switched = send_frame(&mut gate, 5, &[other, challenger]).ok_or("未确认持续出现的新目标")?;
    check(switched["candidate"]["lock_id"] == 5 && switched["candidate"]["bounds"][0] == 195.0,
        "换锁编号或连续候选错误")?;
    Ok(())
}

fn verify_conversion() -> Result<(), String> {
    let mut smoothing = Smoothing::default();
    check(smoothing.relative([3.0, 4.0], [1.0, 1.0], 5.0, 10.0, 24, 1) == [0, 0], "径向死区错误")?;
    check(smoothing.relative([6.0, 10.0], [-0.5, 1.0], 5.0, 10.0, 24, 11) == [-3, 10], "校准方向或初始平滑错误")?;
    check(smoothing.relative([14.0, 20.0], [-0.5, 1.0], 5.0, 10.0, 24, 21) == [-5, 15], "半衰期平滑错误")?;
    check(smoothing.relative([-250.0, 5.0], [-0.5, 1.0], 0.0, 10.0, 24, 31) == [24, 10], "反向或硬限幅错误")?;
    check(smoothing.relative([0.0, 0.0], [1.0, 1.0], 5.0, 10.0, 24, 40) == [0, 0]
        && smoothing.relative([7.0, 0.0], [1.0, 1.0], 5.0, 10.0, 24, 41) == [7, 0], "死区残留了历史命令")?;
    Ok(())
}

fn verify_held_confirmation() -> Result<(), String> {
    let mut gate = setup(MockKmBox::default())?;
    let input_report = |mask, age| {
        let mut event = input(mask, age);
        event["link_at_ms"] = json!(now_ms());
        event
    };
    let confirmation = |mask| json!({"mask":mask,"at_ms":now_ms()});
    check(gate.confirm_buttons(&confirmation(3)).is_none() && gate.target_sample().is_none(),
        "设备 A 的读数单独触发了候选")?;
    gate.buttons(&input_report(3, 1100));
    std::thread::sleep(Duration::from_millis(5));
    let accepted = gate.confirm_buttons(&confirmation(3)).ok_or("真实按键报告触发后未接入确认")?;
    check(accepted["hold_confirmed"] == true, "缺少已确认状态")?;
    std::thread::sleep(Duration::from_millis(85));
    gate.confirm_buttons(&confirmation(3));
    send_frame(&mut gate, 2, &NEXT);
    std::thread::sleep(Duration::from_millis(35));
    let stale = json!({"connected":true,"monitoring":true,"buttons":-1,"age_ms":1220,"link_at_ms":now_ms()});
    check(gate.buttons(&stale).is_none() && gate.tick().is_none() && gate.target_sample().is_some(),
        "已有真实触发和实时确认，但静止时跟随停止")?;
    let release = gate.confirm_buttons(&confirmation(1)).ok_or("读到松手未触发清理")?;
    check(release["candidate"].is_null() && gate.target_sample().is_none(), "松手保留了候选")?;
    std::thread::sleep(Duration::from_millis(3));
    gate.confirm_buttons(&confirmation(3));
    check(gate.target_sample().is_none(), "确认恢复后自行触发")?;

    for (event, reason) in [("confirmation_lost", "设备 A 持键确认或 KMBox 连接应答已过期"),
        ("link_lost", "KMBox 监控已断开或报文过期")] {
        let mut gate = setup(MockKmBox::default())?;
        gate.buttons(&input_report(3, 0));
        gate.confirm_buttons(&confirmation(3));
        let result = if event == "confirmation_lost" {
            gate.confirm_buttons(&json!({"at_ms":0}))
        } else {
            gate.buttons(&json!({"connected":false,"monitoring":false}))
        }.ok_or("失联未锁止")?;
        check(result["state"] == "Fault" && result["candidate"].is_null() && result["reason"] == reason,
            "失联时未清理候选")?;
    }
    Ok(())
}

#[cfg(test)]
fn check_event_release(mask: u8, release: u8, hold_ms: u64) {
    let mut gate = setup(MockKmBox::default()).unwrap();
    let profile = json!({"mask":mask, "event_preview":true, "enabled_classes":[1]});
    gate.configure(vec![serde_json::from_value(profile).unwrap()]).unwrap();
    gate.arm().unwrap();
    let start = Instant::now();
    let mut press = input(mask, 0);
    press["link_at_ms"] = json!(now_ms());
    assert!(gate.buttons(&press).unwrap()["candidate"].is_object());
    let mut sequence = 2;
    let mut marked = false;
    loop {
        let age = start.elapsed().as_millis() as u64;
        let status = json!({"connected":true, "monitoring":true, "buttons":if age > 1200 { -1 } else { mask as i32 },
            "age_ms":age, "reports":1,"report_at_ms":now_ms()-age,"link_at_ms":now_ms()});
        gate.buttons(&status);
        if let Some(update) = send_frame(&mut gate, sequence, &FIRST) {
            marked |= update["event_hold_age_ms"].as_u64().is_some_and(|ms| ms >= 1200);
        }
        gate.tick();
        assert!(gate.target_sample().is_some());
        assert!(gate.movement.driver.sent.is_empty());
        if age >= hold_ms { break; }
        sequence += 1;
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(marked);
    let mut up = input(release, 0);
    up["link_at_ms"] = json!(now_ms());
    let update = gate.buttons(&up).unwrap();
    assert!(update["candidate"].is_null() && update["active"] == false);
    send_frame(&mut gate, sequence + 1, &FIRST);
    assert!(gate.target_sample().is_none());
}

#[test]
fn right_release_after_6100_ms() { check_event_release(2, 0, 6100); }

#[test]
fn chord_release_after_6200_ms() { check_event_release(3, 2, 6200); }

#[test]
fn event_preview_auto_arms_without_mouse_reports_and_rearms_after_release() {
    let mut gate = AutomationController::with_driver(MockKmBox::default());
    let profile = json!({"mask":2, "event_preview":true, "enabled_classes":[1]});
    gate.configure(vec![serde_json::from_value(profile).unwrap()]).unwrap();
    gate.request_model(1);
    gate.request_stream(1);
    gate.worker(&json!({"kind":"model","model":1,"provider":"TensorRT"}));
    gate.worker(&json!({"kind":"stream_state","state":"live","source":1,"frame_at":now_ms()}));
    send_frame(&mut gate, 1, &FIRST);
    let mut status = json!({"connected":true,"monitoring":true,"buttons":-1,"age_ms":-1,"reports":0,"report_at_ms":0});
    assert!(gate.arm().is_err());
    status["link_at_ms"] = json!(now_ms());
    let ready = gate.buttons(&status).unwrap();
    assert_eq!(ready["state"], "Armed");
    assert!(ready["candidate"].is_null());

    let first_press = gate.buttons(&input(2, 0)).unwrap();
    assert!(first_press["candidate"].is_object());
    let first_release = gate.buttons(&input(0, 0)).unwrap();
    assert_eq!(first_release["state"], "Armed");
    assert!(first_release["candidate"].is_null());

    std::thread::sleep(Duration::from_millis(1250));
    status["link_at_ms"] = json!(now_ms());
    gate.buttons(&status);
    send_frame(&mut gate, 2, &FIRST);
    assert!(gate.tick().is_none());
    assert!(gate.buttons(&input(2, 0)).unwrap()["candidate"].is_object());
    assert!(gate.movement.driver.sent.is_empty());

    gate.disarm("手动停止 Dry Run");
    gate.buttons(&input(0, 0));
    gate.buttons(&input(2, 0));
    assert!(gate.target_sample().is_none());
    let profile = json!({"mask":2, "event_preview":true, "enabled_classes":[1]});
    let configured = gate.configure(vec![serde_json::from_value(profile).unwrap()]).unwrap();
    assert_eq!(configured["state"], "Ready");
}

#[cfg(test)]
fn physical_input(mask: u8, report: u64) -> Value {
    let stamp = now_ms();
    let mut event = input(mask, 0);
    event["reports"] = json!(report);
    event["report_at_ms"] = json!(stamp);
    event["link_at_ms"] = json!(stamp);
    event
}

#[cfg(test)]
fn setup_real() -> AutomationController<MockKmBox> {
    let mut gate = setup(MockKmBox::default()).unwrap();
    gate.request_device("1234ABCD");
    gate.request_named_stream(1, "test-source".into());
    gate.worker(&json!({"kind":"stream_state","state":"live","source":1,"frame_at":now_ms()}));
    let profile = json!({"mask":3,"enabled_classes":[1],"classes":[1],"event_preview":true,
        "calibration_site":{"source":"test-source","device":"1234ABCD","desktop":[2560,1440],"crop":[640,640],"observed":[10,10]}});
    gate.configure(vec![serde_json::from_value(profile).unwrap()]).unwrap();
    send_frame(&mut gate, 1, &FIRST);
    assert!(gate.arm_real(1, false).is_err());
    gate.buttons(&physical_input(0, 1));
    assert!(gate.arm_real(1, true).is_err());
    assert_eq!(gate.arm_real(1, false).unwrap()["mode"], "real");
    assert!(gate.movement.driver.live);
    gate
}

#[test]
fn real_tracking_requires_fresh_chord_frames_and_stops_on_release() {
    let mut gate = setup_real();
    gate.buttons(&physical_input(1, 2));
    send_frame(&mut gate, 2, &FIRST);
    assert!(gate.target_sample().is_none());
    gate.buttons(&physical_input(3, 3));
    assert!(gate.target_sample().is_none());
    send_frame(&mut gate, 3, &FIRST);
    assert!(gate.target_sample().is_none());
    for sequence in 4..10 {
        send_frame(&mut gate, sequence, &FIRST);
        gate.tick();
        std::thread::sleep(Duration::from_millis(18));
    }
    let sent = gate.movement.driver.sent.len();
    assert!(sent > 0);
    assert!(gate.movement.driver.sent.iter().all(|m| f64::from(m.dx).hypot(f64::from(m.dy)) <= 2.04));
    gate.tick();
    assert_eq!(gate.movement.driver.sent.len(), sent);
    send_frame(&mut gate, 10, &FIRST);
    let released = gate.buttons(&physical_input(2, 4)).unwrap();
    assert_eq!(released["state"], "Stopped");
    gate.buttons(&physical_input(3, 5));
    gate.tick();
    assert!(!gate.movement.driver.live);
    assert_eq!(gate.movement.driver.sent.len(), sent);
}

#[test]
fn real_tracking_faults_clear_candidates_and_require_new_arm() {
    for fault in ["target", "network", "frame", "input", "model", "ack", "stop"] {
        let mut gate = setup_real();
        gate.buttons(&physical_input(3, 2));
        send_frame(&mut gate, 2, &FIRST);
        send_frame(&mut gate, 3, &FIRST);
        assert!(gate.target_sample().is_some());
        let update = match fault {
            "target" => send_frame(&mut gate, 4, &[]).unwrap(),
            "network" => gate.buttons(&json!({"connected":false,"monitoring":false})).unwrap(),
            "frame" => gate.worker(&json!({"kind":"stream_state","state":"lost","source":1})).unwrap(),
            "input" => gate.buttons(&input(3, 1201)).unwrap(),
            "model" => gate.worker(&json!({"kind":"model","model":1,"error":"fault"})).unwrap(),
            "ack" => gate.live_fault("KMBox 命令应答超时").unwrap(),
            "stop" => gate.disarm("手动停止"),
            _ => unreachable!(),
        };
        assert!(update["candidate"].is_null() && update["armed"] == false, "{fault}");
        assert!(!gate.movement.driver.live, "{fault}");
        gate.buttons(&physical_input(3, 4));
        assert!(gate.target_sample().is_none(), "{fault}");
    }
}

pub(super) fn verify() -> Result<Value, String> {
    verify_conversion()?;
    verify_tracking()?;
    verify_held_confirmation()?;

    let mut calibrated = setup(MockKmBox::default())?;
    let measurement = serde_json::to_value(crate::calibration::estimate(13.0, 13.0)?)
        .map_err(|error| error.to_string())?;
    calibrated.configure(vec![serde_json::from_value(json!({
        "name":"实测校准复核", "mask":3, "enabled_classes":[6],
        "gain_x":measurement["gain_x"], "gain_y":measurement["gain_y"],
        "invert_x":measurement["invert_x"], "invert_y":measurement["invert_y"],
    })).map_err(|error| error.to_string())?])?;
    send_frame(&mut calibrated, 2, &[[287.3, 321.8, 307.3, 341.8, 0.966, 6.0]]);
    calibrated.arm()?;
    let target = calibrated.buttons(&input(3, 0)).ok_or("现场坐标复核未生成候选")?;
    check(target["candidate"]["relative"] == json!([-17,9]) && target["candidate"]["signed_scale"] == json!([10.0/13.0,10.0/13.0]),
        "现场坐标的校准换算或比例记录错误")?;

    let mut gate = setup(MockKmBox::default())?;
    for (single, other) in [(1, 2), (2, 1)] {
        gate.buttons(&input(single, 0));
        gate.tick();
        check(gate.movement.driver.sent.is_empty(), "单键错误调度")?;
        gate.buttons(&input(3, 0));
        gate.buttons(&input(other, 0));
        gate.tick();
        check(gate.movement.driver.sent.is_empty(), "释放后仍发送已排队位移")?;
        gate.buttons(&input(0, 0));
    }
    gate.buttons(&input(3, 0));
    send_frame(&mut gate, 2, &NEXT);
    let at = Instant::now();
    gate.tick_at(at);
    gate.tick_at(at + Duration::from_millis(4));
    check(gate.movement.driver.sent.as_slice() == [RelativeMove { dx: 24, dy: 24 }], "新命令未覆盖旧命令或发生重复提交")?;
    send_frame(&mut gate, 3, &FIRST);
    gate.tick_at(at + Duration::from_millis(8));
    check(gate.movement.submitted == 1, "超出调度频率")?;
    gate.tick_at(at + Duration::from_millis(20));
    check(gate.movement.driver.sent.last() == Some(&RelativeMove { dx: 20, dy: 20 }) && gate.movement.submitted == 2,
        "调度未消费限速期间的最新命令")?;
    send_frame(&mut gate, 4, &NEXT);
    gate.tick_at(Instant::now() + Duration::from_millis(46));
    check(gate.movement.submitted == 2 && gate.movement.expired == 1, "过期命令未丢弃")?;
    send_frame(&mut gate, 5, &NEXT);
    send_frame(&mut gate, 6, &[]);
    gate.tick_at(Instant::now() + Duration::from_millis(40));
    check(gate.movement.submitted == 2, "目标消失后仍提交")?;

    for event in ["network", "stale_buttons", "stale_video", "model", "emergency", "send_failure"] {
        let mut gate = setup(MockKmBox::default())?;
        gate.buttons(&input(3, 0));
        let fault = match event {
            "network" => gate.buttons(&json!({"connected":false,"monitoring":false})),
            "stale_buttons" => gate.buttons(&input(3, 1201)),
            "stale_video" => {
                gate.worker(&json!({"kind":"stream_state","state":"live","source":1,"frame_at":now_ms()-151}));
                gate.tick()
            }
            "model" => gate.worker(&json!({"kind":"model","model":1,"error":"mock"})),
            "emergency" => Some(gate.disarm("mock emergency stop")),
            _ => { gate.movement.driver.disconnected = true; gate.tick() }
        }.ok_or(format!("{event} 未锁止"))?;
        check(fault["state"] == if event == "emergency" { "Stopped" } else { "Fault" }
            && fault["candidate"].is_null(), "故障或急停留有待执行命令")?;
        gate.movement.driver.disconnected = false;
        gate.worker(&json!({"kind":"model","model":1,"provider":"TensorRT"}));
        send_frame(&mut gate, 2, &NEXT);
        gate.buttons(&input(3, 0));
        gate.tick();
        check(gate.movement.driver.sent.is_empty(), "故障恢复后自行武装")?;
        gate.arm()?;
        gate.buttons(&input(3, 0));
        gate.tick();
        check(gate.movement.driver.sent.is_empty(), "人工重新武装时沿用已按住的组合")?;
        gate.buttons(&input(0, 0));
        gate.buttons(&input(3, 0));
        gate.tick();
        check(gate.movement.submitted == 1, "人工重新武装并重新按下后未恢复")?;
    }

    let mut gate = setup(MockKmBox::default())?;
    for sequence in 2..102 {
        gate.buttons(&input(0, 0));
        send_frame(&mut gate, sequence, &FIRST);
        gate.arm()?;
        gate.buttons(&input(3, 0));
        gate.disarm("mock rapid stop");
        gate.tick();
    }
    check(gate.movement.driver.sent.is_empty(), "快速启停后提交了旧命令")?;
    let mut gate = setup(MockKmBox::default())?;
    let physical = |buttons, report| {
        let mut event = input(buttons, 0);
        event["reports"] = json!(report);
        event["report_at_ms"] = json!(now_ms());
        event
    };
    gate.buttons(&physical(3, 10));
    gate.buttons(&physical(1, 11));
    gate.buttons(&physical(3, 10));
    gate.tick();
    check(gate.movement.driver.sent.is_empty(), "乱序报文重新激活已松开的组合")?;
    gate.buttons(&physical(3, 12));
    gate.tick();
    check(gate.movement.submitted == 1, "新序号的有效按键未恢复")?;
    Ok(json!({"result":"PASS","hardware":"simulated","rapid_start_stop_cycles":100,
        "strict_chord_and_release":true,"stale_input_and_video":true,"loss_and_rearm":true,
        "field_calibration_replayed":{"signed_scale":target["candidate"]["signed_scale"],"relative":target["candidate"]["relative"]},
        "held_confirmation":true,
        "out_of_order_report":true,
        "network_and_driver_failure":true,"emergency_stop":true,"expired_and_duplicate_commands":true,
        "rate_limit_and_latest_only":true,"dead_zone_smoothing_axis_limit":true,"target_switch_confirmation":true}))
}

#[test]
fn mock_transport_regression() { verify().unwrap(); }

pub(super) fn write_transport(report: u64, report_at_ms: u64) -> Result<(), String> {
    let stdout = std::io::stdout();
    let mut gate = setup(KmBoxPipe(stdout.lock()))?;
    gate.buttons(&input(1, 0));
    gate.tick();
    let mut chord = input(3, 0);
    chord["reports"] = json!(report);
    chord["report_at_ms"] = json!(report_at_ms);
    gate.buttons(&chord);
    send_frame(&mut gate, 2, &[[270., 328., 310., 368., 0.9, 1.]]);
    // The original target was lost. Two matching detections confirm the replacement.
    let replacement = [[268., 328., 308., 368., 0.9, 1.]];
    send_frame(&mut gate, 3, &replacement);
    send_frame(&mut gate, 4, &replacement);
    gate.tick();
    gate.buttons(&input(2, 0));
    gate.tick();
    check(gate.movement.submitted == 1, "模拟传输测试必须产生且只产生一条相对位移")
}
