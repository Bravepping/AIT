#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod calibration;
mod control;
mod diagnostics;
mod movement;
mod tracking;
mod validation;

use control::{AutomationController, Profile};
use movement::{MonitorWriter, RuntimeDriver};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    env, fs,
    io::{BufRead, BufReader, Write},
    path::{self, Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::{atomic::{AtomicU64, Ordering}, Arc, Mutex},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tauri::{ipc::Channel, AppHandle, Manager, State};

const MODEL_NAME: &str = "1013v26s_640.onnx";

#[derive(Serialize, Deserialize)]
enum Provider {
    Auto,
    CPU,
    CUDA,
    TensorRT,
    #[serde(rename = "TensorRT for RTX")]
    TensorRtForRtx,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
enum WorkerCommand {
    Discover,
    Stream { source: u32, name: String, url: String },
    Model { model: u32, provider: Provider, filename: Option<String> },
    Preview { fps: u8, preview: u32 },
    CudaCheck,
}

struct Worker {
    child: Child,
    stdin: ChildStdin,
    model_path: PathBuf,
}

impl Drop for Worker {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_some() {
            return;
        }
        let _ = self.stdin.write_all(b"{\"cmd\":\"exit\"}\n");
        let deadline = Instant::now() + Duration::from_millis(850);
        while Instant::now() < deadline {
            if self.child.try_wait().ok().flatten().is_some() {
                return;
            }
            thread::sleep(Duration::from_millis(25));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

struct MonitorWorker {
    child: Child,
    writer: MonitorWriter,
}

impl Drop for MonitorWorker {
    fn drop(&mut self) {
        let _ = self.writer.message(&json!({"cmd":"exit"}));
        let deadline = Instant::now() + Duration::from_millis(2500);
        while Instant::now() < deadline {
            if self.child.try_wait().ok().flatten().is_some() { return; }
            thread::sleep(Duration::from_millis(25));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[derive(Default)]
struct WorkerState(Mutex<Option<Worker>>);

#[derive(Default)]
struct MonitorState(Mutex<KmBoxSession>);

#[derive(Default)]
struct KmBoxSession {
    monitor: Option<MonitorWorker>,
    acceptance: Option<Child>,
    generation: Arc<AtomicU64>,
}

impl KmBoxSession {
    fn acceptance_running(&mut self) -> Result<bool, String> {
        let Some(child) = self.acceptance.as_mut() else { return Ok(false); };
        if child.try_wait().map_err(|error| format!("无法确认验收窗口状态，{error}"))?.is_none() {
            return Ok(true);
        }
        self.acceptance = None;
        Ok(false)
    }
}

struct StartupCheck(Option<PathBuf>);

#[derive(Clone)]
struct DryRunState {
    gate: Arc<Mutex<AutomationController<RuntimeDriver>>>,
    events: Arc<Mutex<Option<Channel<Value>>>>,
    generation: Arc<AtomicU64>,
}

impl Default for DryRunState {
    fn default() -> Self {
        Self { gate: Arc::new(Mutex::new(AutomationController::with_driver(RuntimeDriver::default()))),
            events: Arc::default(), generation: Arc::default() }
    }
}

impl DryRunState {
    fn event(&self, event: Value) {
        if let Ok(channel) = self.events.lock() {
            if let Some(channel) = channel.as_ref() {
                let _ = channel.send(event);
            }
        }
    }

    fn update(&self, update: impl FnOnce(&mut AutomationController<RuntimeDriver>) -> Value) {
        if let Ok(mut gate) = self.gate.lock() {
            self.event(update(&mut gate));
        }
    }
}

fn hidden_process(path: &Path) -> Command {
    let mut command = Command::new(path);
    command.stdin(Stdio::piped()).stdout(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    command
}

fn has_runtime(root: &Path) -> bool {
    root.join("vision-runtime/python.exe").is_file()
        && root.join("vision-runtime/worker.py").is_file()
        && root.join(MODEL_NAME).is_file()
}

fn find_package(app: &AppHandle) -> Result<PathBuf, String> {
    if let Some(override_dir) = env::var_os("VISION_PACKAGE_DIR") {
        let root = PathBuf::from(override_dir);
        return path::absolute(root)
            .ok()
            .filter(|path| has_runtime(path))
            .ok_or_else(|| "VISION_PACKAGE_DIR 中没有可用的推理运行库和模型".into());
    }

    let executable = env::current_exe().map_err(|error| error.to_string())?;
    let resources = app.path().resource_dir().ok();
    executable
        .parent()
        .map(Path::to_path_buf)
        .into_iter()
        .chain(resources)
        .find(|path| has_runtime(path))
        .ok_or_else(|| "请完整解压 V1.1 程序，确认 vision-runtime 和模型文件在同一目录".into())
}

fn model_files(root: &Path) -> Result<Vec<String>, String> {
    let mut models: Vec<String> = fs::read_dir(root).map_err(|error| error.to_string())?
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_file()))
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext.eq_ignore_ascii_case("onnx")))
        .filter_map(|entry| entry.file_name().into_string().ok())
        .collect();
    models.sort_unstable();
    Ok(models)
}

#[tauri::command]
fn list_models(app: AppHandle) -> Result<Vec<String>, String> {
    model_files(&find_package(&app)?)
}

#[tauri::command]
fn start_worker(
    app: AppHandle,
    state: State<WorkerState>,
    dry_run: State<DryRunState>,
    mode: State<StartupCheck>,
    events: Channel<Value>,
) -> Result<(), String> {
    let mut slot = state.0.lock().map_err(|error| error.to_string())?;
    let generation = dry_run.generation.fetch_add(1, Ordering::SeqCst) + 1;
    *dry_run.events.lock().map_err(|error| error.to_string())? = Some(events.clone());
    dry_run.update(AutomationController::restart);
    drop(slot.take());

    let root = find_package(&app)?;
    let runtime = root.join("vision-runtime");
    let mut command = hidden_process(&runtime.join("python.exe"));
    command
        .args(["-B", "-X", "utf8", "-u"])
        .arg(runtime.join("worker.py"))
        .arg("--desktop")
        .current_dir(&root)
        .stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .map_err(|error| format!("无法启动推理进程，{error}"))?;
    let stdin = child.stdin.take().expect("piped worker stdin");
    let stdout = child.stdout.take().expect("piped worker stdout");
    let stderr = child.stderr.take().expect("piped worker stderr");
    *slot = Some(Worker { child, stdin, model_path: root.join(MODEL_NAME) });
    let startup = mode.0.clone();
    if let Some(path) = startup.as_ref() {
        let _ = fs::write(path.with_extension("phase"), "WORKER_SPAWNED");
    }
    let output_events = events.clone();
    let safety = dry_run.inner().clone();
    thread::spawn(move || {
        let mut diagnostics = diagnostics::StreamDiagnostics::default();
        for line in BufReader::new(stdout).split(b'\n') {
            let Ok(line) = line else { break };
            if safety.generation.load(Ordering::SeqCst) != generation {
                return;
            }
            if line.is_empty() {
                continue;
            }
            let payload = serde_json::from_slice(&line).unwrap_or_else(|_| {
                json!({"kind": "log", "text": String::from_utf8_lossy(&line)})
            });
            if payload["kind"] == "ready" {
                if let Some(path) = startup.as_ref() {
                    let _ = fs::write(path.with_extension("phase"), "WORKER_READY");
                }
            }
            if let Ok(mut gate) = safety.gate.lock() {
                if safety.generation.load(Ordering::SeqCst) != generation {
                    return;
                }
                if let Some(update) = gate.worker(&payload) {
                    safety.event(update);
                }
                if let Some(note) = diagnostics.observe(&payload, gate.source_revision(), gate.target_update()) {
                    safety.event(note);
                }
            }
            if payload["kind"] == "detections" || payload["kind"] == "ndi_health" {
                continue;
            }
            if output_events.send(payload).is_err() {
                return;
            }
        }
        if safety.generation.load(Ordering::SeqCst) == generation {
            safety.update(|controller| controller.worker(&json!({"kind":"worker_exit"})).expect("worker exit state"));
            let _ = output_events.send(json!({"kind": "worker_exit"}));
        }
    });
    thread::spawn(move || {
        for line in BufReader::new(stderr).split(b'\n') {
            let Ok(line) = line else { break };
            let text = String::from_utf8_lossy(&line);
            if !text.trim().is_empty()
                && events.send(json!({"kind": "log", "text": text.trim()})).is_err()
            {
                return;
            }
        }
    });
    Ok(())
}

#[tauri::command]
fn send_worker(state: State<WorkerState>, dry_run: State<DryRunState>, command: WorkerCommand) -> Result<(), String> {
    let mut slot = state.0.lock().map_err(|error| error.to_string())?;
    match &command {
        WorkerCommand::Model { model, .. } => dry_run.update(|gate| gate.request_model(*model)),
        WorkerCommand::Stream { source, name, .. } => dry_run.update(|gate| gate.request_named_stream(*source, name.clone())),
        _ => {}
    }
    let worker = slot.as_mut().ok_or("推理进程尚未启动")?;
    let mut payload = serde_json::to_value(command).map_err(|error| error.to_string())?;
    if payload["cmd"] == "model" {
        let filename = payload.get("filename").and_then(Value::as_str).unwrap_or(MODEL_NAME);
        let root = worker.model_path.parent().ok_or("未定位模型目录")?;
        if !model_files(root)?.iter().any(|item| item == filename) {
            return Err("所选 ONNX 文件已移除，请刷新模型列表".into());
        }
        payload["path"] = json!(root.join(filename));
    }

    serde_json::to_writer(&mut worker.stdin, &payload).map_err(|error| error.to_string())?;
    worker
        .stdin
        .write_all(b"\n")
        .and_then(|_| worker.stdin.flush())
        .map_err(|error| format!("无法向推理进程发送命令，{error}"))
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
enum MonitorCommand {
    Connect { ip: String, port: u16, uuid: String },
    Monitor { port: u16 },
    StopMonitor,
    TraceWire,
    Disconnect,
    ConfirmInput { pairing: String },
    StopConfirmation,
}

#[tauri::command]
fn send_monitor(app: AppHandle, state: State<MonitorState>, dry_run: State<DryRunState>, command: MonitorCommand) -> Result<(), String> {
    let mut session = state.0.lock().map_err(|error| error.to_string())?;
    if session.acceptance_running()? { return Err("请先关闭实体鼠标验收窗口，再连接主程序的只读监控".into()); }
    if matches!(&command, MonitorCommand::ConfirmInput { .. } | MonitorCommand::StopConfirmation) {
        if cfg!(feature = "kmbox-hold-diagnostics") { return Err("本测试包只使用 KMBox NET 原始五键报文".into()); }
        dry_run.update(AutomationController::confirmation_changed);
    } else if let MonitorCommand::Connect { uuid, .. } = &command {
        dry_run.update(|gate| {
            let update = gate.request_device(uuid);
            gate.movement.driver.reset_connection();
            update
        });
    } else if !matches!(&command, MonitorCommand::TraceWire) {
        dry_run.update(AutomationController::monitor_changed);
    }
    let generation = session.generation.clone();
    let slot = &mut session.monitor;
    if slot.as_mut().is_some_and(|worker| worker.child.try_wait().ok().flatten().is_some()) {
        drop(slot.take());
    }
    if slot.is_none() {
        if !matches!(command, MonitorCommand::Connect { .. }) {
            return Err("请先连接 KMBox".into());
        }
        let path = find_package(&app)?.join("VisionKmMonitor.exe");
        let mut child = hidden_process(&path)
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| format!("无法启动 KMBox 只读监控，{error}"))?;
        let stdin = child.stdin.take().expect("piped monitor stdin");
        let stdout = child.stdout.take().expect("piped monitor stdout");
        let writer = MonitorWriter(Arc::new(Mutex::new(stdin)));
        dry_run.gate.lock().map_err(|error| error.to_string())?.movement.driver.pipe = Some(writer.clone());
        *slot = Some(MonitorWorker { child, writer });
        let current = generation.fetch_add(1, Ordering::SeqCst) + 1;
        let safety = dry_run.inner().clone();
        thread::spawn(move || {
            for line in BufReader::new(stdout).split(b'\n') {
                let Ok(line) = line else { break };
                let Ok(event) = serde_json::from_slice::<Value>(&line) else { continue };
                if generation.load(Ordering::SeqCst) != current { return; }
                if event["kind"] == "kmbox_confirmation" {
                    if let Ok(mut gate) = safety.gate.lock() {
                        if generation.load(Ordering::SeqCst) != current { return; }
                        if let Some(update) = gate.confirm_buttons(&event) { safety.event(update); }
                    }
                    continue;
                }
                if event["kind"] == "kmbox" {
                    if let Ok(mut gate) = safety.gate.lock() {
                        if generation.load(Ordering::SeqCst) != current { return; }
                        gate.movement.driver.observe_axes(event["xy_status"].as_str(), event["busy"] == true);
                        if let Some(update) = gate.buttons(&event) {
                            safety.event(update);
                        }
                    }
                }
                if event["kind"] == "auto_takeover_status" {
                    if let Ok(mut gate) = safety.gate.lock() {
                        if generation.load(Ordering::SeqCst) != current { return; }
                        let id = event["id"].as_u64().unwrap_or_default();
                        let result = gate.movement.driver.takeover_result(id, event["status"] == "acknowledged");
                        if result == Some(false) {
                            if let Some(update) = gate.live_fault("KMBox 未确认接管，请重新武装") { safety.event(update); }
                        }
                    }
                    continue;
                }
                if event["kind"] == "move_status" {
                    if let Ok(mut gate) = safety.gate.lock() {
                        let id = event["id"].as_u64().unwrap_or_default();
                        if let Some((target, elapsed, queue_ms)) = gate.movement.driver.result(id) {
                            let (target, class, frame_at) = target.unwrap_or_default();
                            safety.event(json!({"kind":"move_receipt", "id":id, "target":target, "class":class,
                                "dx":event["dx"], "dy":event["dy"], "status":event["status"],
                                "queue_ms":queue_ms, "exchange_ms":event["elapsed_ms"], "dispatch_ms":elapsed.as_secs_f64()*1000.0,
                                "receiver_ms":control::now_ms().saturating_sub(frame_at)}));
                            if event["status"] != "acknowledged" {
                                if let Some(update) = gate.live_fault("KMBox 位移未确认，请重新武装") { safety.event(update); }
                            }
                        }
                    }
                    continue;
                }
                if event["kind"] == "auto_unavailable" {
                    if let Ok(mut gate) = safety.gate.lock() {
                        if let Some(update) = gate.live_fault("KMBox 拒绝武装，请检查物理按键和连接") { safety.event(update); }
                    }
                    continue;
                }
                safety.event(event);
            }
            if let Ok(mut gate) = safety.gate.lock() {
                if generation.load(Ordering::SeqCst) == current {
                    let recovery_pending = gate.movement.driver.axes_need_recovery();
                    gate.movement.driver.detach();
                    safety.event(gate.monitor_failed());
                    safety.event(json!({"kind":"kmbox_exit", "recovery_pending":recovery_pending}));
                }
            };
        });
    }
    let worker = slot.as_mut().ok_or("监控进程尚未启动")?;
    worker.writer.message(&command).map_err(|error| error.to_string())
}

#[tauri::command]
fn open_kmbox_acceptance(state: State<MonitorState>, dry_run: State<DryRunState>) -> Result<(), String> {
    // Resolve only the sibling shipped with this app. The inference override does not select executables here.
    let executable = env::current_exe().map_err(|error| error.to_string())?;
    let root = executable.parent().ok_or("无法定位当前程序目录")?;
    let path = root.join("VisionKmCalibration.exe");
    if !path.is_file() { return Err("当前目录缺少 VisionKmCalibration.exe，请完整解压现场验收包".into()); }
    let mut session = state.0.lock().map_err(|error| error.to_string())?;
    if session.acceptance_running()? { return Err("实体鼠标验收窗口已打开".into()); }
    {
        let mut gate = dry_run.gate.lock().map_err(|error| error.to_string())?;
        if gate.movement.driver.axes_need_recovery() {
            return Err("请先停止真实跟随并确认物理 X/Y 恢复，再打开校准工具".into());
        }
        session.generation.fetch_add(1, Ordering::SeqCst);
        gate.monitor_changed();
        gate.movement.driver.detach();
        dry_run.event(gate.disarm("实体鼠标验收期间，Dry Run 停止"));
    }
    drop(session.monitor.take());
    session.acceptance = Some(Command::new(&path).current_dir(root).spawn()
        .map_err(|error| format!("实体鼠标验收窗口启动失败，{error}"))?);
    Ok(())
}

#[tauri::command]
fn configure_gate(state: State<DryRunState>, profiles: Vec<Profile>) -> Result<(), String> {
    let mut gate = state.gate.lock().map_err(|error| error.to_string())?;
    match gate.configure(profiles) {
        Ok(update) => {
            state.event(update);
            Ok(())
        }
        Err(error) => {
            state.event(gate.configure(Vec::new())?);
            Err(error)
        }
    }
}

#[tauri::command]
fn validate_profile_backup(profiles: Vec<Profile>) -> Result<(), String> {
    AutomationController::<movement::DryRunDriver>::validate_profiles(&profiles)
}

#[tauri::command]
fn arm_gate(state: State<DryRunState>) -> Result<(), String> {
    let mut gate = state.gate.lock().map_err(|error| error.to_string())?;
    let update = gate.arm()?;
    state.event(update);
    Ok(())
}

#[tauri::command]
fn arm_real_tracking(state: State<DryRunState>, profile_id: u32) -> Result<(), String> {
    if cfg!(feature = "kmbox-hold-diagnostics") { return Err("KMBox 长按专项包的真实自动发送已锁定".into()); }
    let mut gate = state.gate.lock().map_err(|error| error.to_string())?;
    state.event(gate.arm_real(profile_id)?);
    Ok(())
}

#[tauri::command]
fn stop_gate(state: State<DryRunState>) {
    state.update(AutomationController::stop_manual);
}

#[tauri::command]
fn estimate_calibration(observed_x: f64, observed_y: f64) -> Result<calibration::Calibration, String> {
    calibration::estimate(observed_x, observed_y)
}

#[tauri::command]
fn report_frontend_ready(app: AppHandle, mode: State<StartupCheck>) -> Result<(), String> {
    if let Some(path) = mode.0.as_ref() {
        fs::write(path, "V1.4_FRONTEND_AND_WORKER_READY").map_err(|error| error.to_string())?;
        app.exit(0);
    }
    Ok(())
}

#[tauri::command]
fn export_log(app: AppHandle, text: String) -> Result<String, String> {
    let folder = app.path().app_log_dir().map_err(|error| error.to_string())?;
    fs::create_dir_all(&folder).map_err(|error| error.to_string())?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_millis();
    let path = folder.join(format!("vision-{timestamp}.txt"));
    fs::write(&path, text).map_err(|error| error.to_string())?;
    Ok(path.display().to_string())
}

fn main() {
    let mut args = env::args_os().skip(1);
    let startup = match args.next().as_deref().and_then(std::ffi::OsStr::to_str) {
        Some("--verify-start") => args.next().map(PathBuf::from),
        Some("--verify-control-stream") => {
            if let Err(error) = validation::verify_control_stream() {
                eprintln!("{error}");
                std::process::exit(1);
            }
            return;
        }
        Some("--verify-physical-dry-run") => {
            if let Err(error) = validation::verify_physical_dry_run() {
                eprintln!("{error}");
                std::process::exit(1);
            }
            return;
        }
        Some("--verify-kmbox-commands") => {
            let mut numbers = args.map(|arg| arg.to_string_lossy().parse::<u64>().ok());
            let result = numbers.next().flatten().zip(numbers.next().flatten())
                .ok_or_else(|| "expected simulated report number and timestamp".into())
                .and_then(|(report, stamp)| validation::verify_transport_stream(report, stamp));
            if let Err(error) = result {
                eprintln!("{error}");
                std::process::exit(1);
            }
            return;
        }
        Some("--verify-tracking-replay") => {
            let result = args.next().zip(args.next()).ok_or_else(|| "expected replay input and output".to_string())
                .and_then(|(input, output)| {
                    let bytes = fs::read(input).map_err(|e| e.to_string())?;
                    let input = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
                    let report = control::replay_tracking(&input)?;
                    fs::write(output, serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?)
                        .map_err(|e| e.to_string())
                });
            if let Err(error) = result {
                if let Ok(executable) = env::current_exe() {
                    let _ = fs::write(executable.with_extension("test-error.txt"), error);
                }
                std::process::exit(1);
            }
            return;
        }
        Some("--verify-dry-run") => {
            let result = args.next().zip(args.next()).ok_or_else(|| "expected input and output".into())
                .and_then(|(input, output)| validation::verify_trace(Path::new(&input), Path::new(&output)));
            if let Err(error) = result {
                if let Ok(path) = env::current_exe().map(|path| path.with_extension("test-error.txt")) {
                    let _ = fs::write(path, &error);
                }
                std::process::exit(1);
            }
            return;
        }
        Some(mode @ ("--verify-live-dry-run" | "--verify-ndi-watchdog")) => {
            let pause = if mode == "--verify-ndi-watchdog" {
                args.next().map(PathBuf::from)
            } else { None };
            let result = args.next().zip(args.next()).ok_or_else(|| "expected source and output".into())
                .and_then(|(source, output)| validation::verify_live(&source.to_string_lossy(), Path::new(&output), pause.as_deref()));
            if let Err(error) = result {
                if let Ok(path) = env::current_exe().map(|path| path.with_extension("test-error.txt")) {
                    let _ = fs::write(path, &error);
                }
                std::process::exit(1);
            }
            return;
        }
        _ => None,
    };
    let dry_run = DryRunState::default();
    let watchdog = dry_run.clone();
    thread::spawn(move || loop {
        thread::sleep(Duration::from_millis(10));
        if let Ok(mut gate) = watchdog.gate.lock() {
            if gate.movement.driver.timed_out() {
                if let Some(update) = gate.live_fault("KMBox 命令应答超时，请重新连接并武装") {
                    watchdog.event(update);
                }
            }
            if let Some(update) = gate.tick() {
                watchdog.event(update);
            }
        }
    });
    let app = tauri::Builder::default()
        .manage(WorkerState::default())
        .manage(MonitorState::default())
        .manage(dry_run)
        .manage(StartupCheck(startup))
        .setup(|app| {
            if let Some(path) = app.state::<StartupCheck>().0.as_ref() {
                let _ = fs::write(path.with_extension("phase"), "WEBVIEW_INITIALIZED");
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            start_worker, send_worker, send_monitor, open_kmbox_acceptance, configure_gate, validate_profile_backup, arm_gate, arm_real_tracking, stop_gate, estimate_calibration, list_models, report_frontend_ready, export_log
        ])
        .build(tauri::generate_context!())
        .expect("build NDI Vision Control");
    app.run(|app, event| {
        if matches!(event, tauri::RunEvent::ExitRequested { .. }) {
            app.state::<DryRunState>().update(|gate| gate.disarm("程序退出"));
            if let Ok(mut slot) = app.state::<MonitorState>().0.lock() {
                drop(slot.monitor.take());
            }
            if let Ok(mut slot) = app.state::<WorkerState>().0.lock() {
                drop(slot.take());
            }
        }
    });
}
