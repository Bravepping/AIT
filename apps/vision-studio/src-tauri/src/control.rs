use crate::movement::{Dispatcher, DryRunDriver, HeldInput, KmBoxDriver, MotionConfig, Smoothing, KMBOX_AXIS_LIMIT};
use crate::tracking::{Observation, Tracker};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashSet},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const BUTTON_TIMEOUT: Duration = Duration::from_millis(1200);
const CONFIRMATION_TIMEOUT_MS: u64 = 150;
const LINK_CONFIRMATION_TIMEOUT_MS: u64 = 1200;
pub(crate) const FRAME_TIMEOUT_MS: u64 = 150;

#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Priority {
    #[default]
    Closest,
    Confidence,
    Largest,
}

#[derive(Deserialize)]
#[serde(default)]
pub struct Profile {
    id: u32,
    name: String,
    mask: u8,
    toggle: bool,
    event_preview: bool,
    enabled_classes: Vec<u16>,
    classes: Option<Vec<u16>>,
    priority: Priority,
    confidence: f64,
    anchor_x: f64,
    anchor_y: f64,
    reference_x: Option<f64>,
    reference_y: Option<f64>,
    range_percent: f64,
    gain_x: f64,
    gain_y: f64,
    invert_x: bool,
    invert_y: bool,
    max_step: i32,
    dead_zone_px: f64,
    smooth_half_life_ms: f64,
    switch_confirm_frames: u8,
    per_class: BTreeMap<u16, ClassRule>,
    dead_zone_axes: Option<[f64; 2]>,
    max_speed: u16,
    acceleration: u16,
    tracking_gain: [f64; 2],
    calibration_site: Option<CalibrationSite>,
    motion: MotionConfig,
    lost_wait_ms: u64,
    dynamic_range_percent: Option<f64>,
}

#[derive(Deserialize)]
struct ClassRule {
    threshold: f64,
    priority: i16,
    anchor: Option<[f64; 2]>,
}

#[derive(Deserialize)]
struct CalibrationSite {
    source: String,
    device: String,
    desktop: [u32; 2],
    crop: [u32; 2],
    observed: [f64; 2],
}

impl Default for Profile {
    fn default() -> Self {
        Self {
            id: 1, name: "左右键组合".into(), mask: 3, toggle: false, event_preview: false, enabled_classes: Vec::new(),
            classes: None, priority: Priority::Closest, confidence: 0.55, anchor_x: 0.0, anchor_y: 0.0,
            reference_x: None, reference_y: None,
            range_percent: 100.0, gain_x: 1.0, gain_y: 1.0, invert_x: false, invert_y: false,
            max_step: KMBOX_AXIS_LIMIT, dead_zone_px: 0.0, smooth_half_life_ms: 0.0,
            switch_confirm_frames: 2, per_class: BTreeMap::new(), dead_zone_axes: None,
            max_speed: 480, acceleration: 4000, tracking_gain: [0.35, 0.35], calibration_site: None,
            motion: MotionConfig::default(), lost_wait_ms: 0, dynamic_range_percent: None,
        }
    }
}

impl Profile {
    fn validate(&self) -> Result<(), String> {
        if self.id == 0 || self.name.trim().is_empty() || self.name.chars().count() > 36 {
            return Err("请输入有效的热键方案名称".into());
        }
        if self.mask == 0 || self.mask > 31 || (self.mask & 3 == 3 && self.toggle) {
            return Err(format!("{}：请选择物理按键，包含左右键的组合须使用按住模式", self.name));
        }
        if !(0.55..=1.0).contains(&self.confidence)
            || !(-50.0..=50.0).contains(&self.anchor_x) || !(-50.0..=50.0).contains(&self.anchor_y)
            || [self.reference_x, self.reference_y].into_iter().flatten().any(|v| !(0.0..=100.0).contains(&v))
            || !(1.0..=100.0).contains(&self.range_percent)
        {
            return Err(format!("{}：置信度 55–100%，中心偏移 ±50%，有效半径 1–100%", self.name));
        }
        if !(0.01..=10.0).contains(&self.gain_x) || !(0.01..=10.0).contains(&self.gain_y)
            || !(1..=KMBOX_AXIS_LIMIT).contains(&self.max_step)
        {
            return Err("位移比例须在 0.01–10，单轴上限须在 1–24".into());
        }
        if !(1..=8).contains(&self.switch_confirm_frames) || !(0.0..=96.0).contains(&self.dead_zone_px)
            || !(0.0..=150.0).contains(&self.smooth_half_life_ms)
        {
            return Err("换目标确认须在 1–8 帧，死区须在 0–96 px，平滑半衰期须在 0–150 ms".into());
        }
        if self.enabled_classes.len() > 256 || self.classes.as_ref().is_some_and(|ids| ids.len() > 256) {
            return Err("类别数量超出范围".into());
        }
        if self.per_class.len() > 256 || self.per_class.values().any(|rule| !(0.55..=1.0).contains(&rule.threshold))
            || self.per_class.values().filter_map(|rule| rule.anchor).flatten().any(|v| !(-50.0..=50.0).contains(&v))
            || self.dead_zone_axes.is_some_and(|axes| axes.into_iter().any(|n| !(0.0..=96.0).contains(&n)))
            || !(90..=1200).contains(&self.max_speed) || !(1..=20000).contains(&self.acceleration)
            || self.tracking_gain.into_iter().any(|n| !(0.01..=2.0).contains(&n))
        {
            return Err("请检查类别阈值、轴向死区、速度和加速度".into());
        }
        if !self.motion.validate() || self.lost_wait_ms > 500
            || self.dynamic_range_percent.is_some_and(|radius| !(1.0..=self.range_percent).contains(&radius))
        {
            return Err("请检查跟随算法参数、丢失等待（0–500 ms）和动态起始半径".into());
        }
        Ok(())
    }

    fn includes(&self, class: u16) -> bool {
        self.enabled_classes.contains(&class) && self.classes.as_ref().is_none_or(|ids| ids.contains(&class))
    }
}

struct Binding {
    profile: Profile,
    released: bool,
    down: bool,
    latched: bool,
}

impl Binding {
    fn reset(&mut self, buttons: u8) {
        self.down = self.is_down(buttons);
        self.released = !self.down;
        self.latched = false;
    }

    fn is_down(&self, buttons: u8) -> bool {
        buttons & self.profile.mask == self.profile.mask
    }

    fn active(&self) -> bool {
        self.released && if self.profile.toggle { self.latched } else { self.down }
    }

    fn update(&mut self, buttons: u8) {
        let down = self.is_down(buttons);
        if !down {
            self.released = true;
        }
        if self.released && down && !self.down && self.profile.toggle {
            self.latched = !self.latched;
        }
        self.down = down;
    }
}

#[derive(Deserialize)]
struct Frame {
    source: u32,
    model: u32,
    frame_at: u64,
    sequence: u64,
    width: u32,
    height: u32,
    boxes: Vec<[f64; 6]>,
}

struct TargetMatch {
    observation: Observation,
    anchor: [f64; 2],
    distance: f64,
    score: f64,
    rank: f64,
    class_priority: i16,
}

#[derive(Default)]
struct TargetLock {
    tracker: Tracker,
    incumbent: Option<(u64, u64, u64)>,
    missing: bool,
    waiting: bool,
    challenger: Option<(u64, u8, u64)>,
}

#[derive(Serialize)]
struct Candidate {
    profile_id: u32,
    class: u16,
    score: f64,
    bounds: [f64; 4],
    center: [f64; 2],
    anchor: [f64; 2],
    reference: [f64; 2],
    normalized: [f64; 2],
    offset: [f64; 2],
    signed_scale: [f64; 2],
    relative: [i32; 2],
    dimensions: [u32; 2],
    sequence: u64,
    lock_id: u64,
    track_id: u64,
    target_velocity_px_s: [f64; 2],
    range_px: f64,
    latency_ms: u64,
    computed_at_ms: u64,
    decision_ms: f64,
    selection_ms: f64,
    movement_ms: f64,
    selection_reason: &'static str,
}

const NO_TARGET: &str = "没有满足置信度、类别和有效范围的检测框";

impl Frame {
    fn select(&self, profile: &Profile, lock: &mut TargetLock, smoothing: &mut Smoothing) -> Option<Candidate> {
        self.select_with_range(profile, lock, smoothing, f64::INFINITY, [1.0; 2])
    }

    fn select_with_range(&self, profile: &Profile, lock: &mut TargetLock, smoothing: &mut Smoothing,
        max_range_px: f64, tracking_gain: [f64; 2]) -> Option<Candidate> {
        let started = Instant::now();
        let dimensions = [self.width, self.height];
        if dimensions.contains(&0) {
            return None;
        }
        let image = dimensions.map(f64::from);
        let reference = [image[0] * profile.reference_x.unwrap_or(50.0) / 100.0,
            image[1] * profile.reference_y.unwrap_or(50.0) / 100.0];
        let range = (image[0].min(image[1]) * profile.range_percent / 200.0).min(max_range_px);
        let initial_range = profile.dynamic_range_percent.map_or(range, |percent| {
            (image[0].min(image[1]) * percent / 200.0).min(range)
        });
        if lock.missing && lock.incumbent.is_some_and(|(_, _, seen)| {
            self.frame_at.saturating_sub(seen) > profile.lost_wait_ms
        }) {
            lock.incumbent = None;
            lock.missing = false;
        }
        let observations: Vec<_> = self.boxes.iter().filter_map(|row| {
            let [left, top, right, bottom, score, category] = *row;
            if !row.iter().all(|value| value.is_finite())
                || category < 0.0 || category > f64::from(u16::MAX) || category.fract() != 0.0
                || !profile.includes(category as u16)
            {
                return None;
            }
            let class_rule = profile.per_class.get(&(category as u16));
            if score < class_rule.map_or(profile.confidence, |rule| rule.threshold) { return None; }
            let bounds = [left.clamp(0.0, image[0]), top.clamp(0.0, image[1]),
                          right.clamp(0.0, image[0]), bottom.clamp(0.0, image[1])];
            let size = [bounds[2] - bounds[0], bounds[3] - bounds[1]];
            if size[0] <= 0.0 || size[1] <= 0.0 {
                return None;
            }
            let [anchor_x, anchor_y] = class_rule.and_then(|rule| rule.anchor).unwrap_or([profile.anchor_x, profile.anchor_y]);
            let anchor_ratio = [0.5 + anchor_x / 100.0, 0.5 + anchor_y / 100.0];
            let anchor = [bounds[0] + size[0] * anchor_ratio[0], bounds[1] + size[1] * anchor_ratio[1]];
            let distance = (anchor[0] - reference[0]).powi(2) + (anchor[1] - reference[1]).powi(2);
            let observation = Observation { class: category as u16, bounds };
            let rank = match profile.priority {
                Priority::Closest => -distance,
                Priority::Confidence => score,
                Priority::Largest => observation.area(),
            };
            Some(TargetMatch { observation, anchor, distance, score, rank,
                class_priority: class_rule.map_or(0, |rule| rule.priority) })
        }).collect();
        let geometry: Vec<_> = observations.iter().map(|item| item.observation).collect();
        let tracks = lock.tracker.update(&geometry, self.sequence, self.frame_at, profile.lost_wait_ms);
        let selected = observations.into_iter().zip(tracks.iter().copied()).filter_map(|(observation, track)| {
            let radius = if let Some((incumbent, _, _)) = lock.incumbent {
                if track.id != incumbent { return None; }
                let speed = track.velocity[0].hypot(track.velocity[1]);
                initial_range + (range - initial_range) * (speed / 0.6).clamp(0.0, 1.0)
            } else { initial_range };
            (observation.distance <= radius * radius).then_some((observation, track, radius))
        }).max_by(|(a, at, _), (b, bt, _)| {
            let continuing = |id| lock.challenger.is_some_and(|(previous, _, _)| previous == id);
            continuing(at.id).cmp(&continuing(bt.id))
                .then_with(|| a.class_priority.cmp(&b.class_priority))
                .then_with(|| a.rank.total_cmp(&b.rank)).then_with(|| b.distance.total_cmp(&a.distance))
        });
        let Some((TargetMatch { observation, anchor, score, .. }, track, radius)) = selected else {
            lock.challenger = None;
            smoothing.reset();
            if lock.incumbent.is_some() {
                lock.waiting = true;
                lock.missing = true;
                if profile.lost_wait_ms == 0 { lock.incumbent = None; }
            }
            return None;
        };
        let (lock_id, selection_reason) = if let Some((_, lock_id, _)) = lock.incumbent {
            (lock_id, if lock.missing { "恢复同一 Track ID" } else { "延续同一 Track ID" })
        } else if lock.waiting {
            let consecutive = lock.challenger.filter(|(previous, _, _)| *previous == track.id)
                .map_or(1, |(_, count, sequence)| count + u8::from(sequence != self.sequence));
            if consecutive < profile.switch_confirm_frames {
                lock.challenger = Some((track.id, consecutive, self.sequence));
                return None;
            }
            lock.waiting = false;
            lock.challenger = None;
            (self.sequence, "新目标已连续确认")
        } else {
            (self.sequence, match profile.priority {
                Priority::Closest => "距参考点最近",
                Priority::Confidence => "置信度最高",
                Priority::Largest => "检测框面积最大",
            })
        };
        lock.incumbent = Some((track.id, lock_id, self.frame_at));
        lock.missing = false;
        lock.waiting = false;
        let selected_at = Instant::now();
        let offset = [anchor[0] - reference[0], anchor[1] - reference[1]];
        let scale = [profile.gain_x * tracking_gain[0] * if profile.invert_x { -1.0 } else { 1.0 },
                     profile.gain_y * tracking_gain[1] * if profile.invert_y { -1.0 } else { 1.0 }];
        let adjusted = profile.dead_zone_axes.map_or(offset, |zones| std::array::from_fn(|axis| {
            if offset[axis].abs() <= zones[axis] { 0.0 } else { offset[axis] }
        }));
        let radial = if profile.dead_zone_axes.is_some() { 0.0 } else { profile.dead_zone_px };
        let relative = smoothing.controlled(adjusted, track.velocity, scale, radial,
            profile.smooth_half_life_ms, profile.max_step, self.frame_at, &profile.motion);
        let computed_at_ms = now_ms();
        Some(Candidate {
            profile_id: profile.id, class: observation.class, score, bounds: observation.bounds,
            center: observation.center(), anchor, reference,
            normalized: [anchor[0] / image[0], anchor[1] / image[1]], offset, signed_scale: scale, relative, dimensions,
            sequence: self.sequence, lock_id, track_id: track.id,
            target_velocity_px_s: track.velocity.map(|axis| axis * 1000.0), range_px: radius,
            latency_ms: computed_at_ms.saturating_sub(self.frame_at), computed_at_ms,
            decision_ms: started.elapsed().as_secs_f64() * 1000.0,
            selection_ms: selected_at.duration_since(started).as_secs_f64() * 1000.0,
            movement_ms: selected_at.elapsed().as_secs_f64() * 1000.0,
            selection_reason,
        })
    }
}

#[derive(Clone, Copy, Default, Eq, PartialEq, Serialize)]
pub enum AutomationState {
    #[default]
    Idle,
    Ready,
    Armed,
    Running,
    Stopped,
    Fault,
}

#[derive(Serialize)]
struct StopDiagnostic {
    id: u64,
    at_ms: u64,
    reason: &'static str,
    previous_state: AutomationState,
    mode: &'static str,
    video_frame_at_ms: Option<u64>,
    video_age_ms: Option<u64>,
    inference_frame_at_ms: Option<u64>,
    inference_age_ms: Option<u64>,
    age_limit_ms: u64,
    frame_sequence: Option<u64>,
    track_id: Option<u64>,
    button_age_ms: Option<u64>,
    cleared_pending: bool,
}

pub struct AutomationController<D = DryRunDriver> {
    bindings: Vec<Binding>,
    state: AutomationState,
    auto_preview_pending: bool,
    real: Option<RealTracking>,
    trial_acknowledged: bool,
    reason: &'static str,
    active: Option<usize>,
    buttons: u8,
    report_at: Option<Instant>,
    physical_report: Option<(u64, u64)>,
    confirmation: Option<(u8, u64)>,
    confirmation_used: bool,
    link_at: u64,
    source: u32,
    source_name: String,
    device: String,
    model: u32,
    stream_live: bool,
    video_at: Option<u64>,
    model_ready: bool,
    frame: Option<Frame>,
    lock: TargetLock,
    smoothing: Smoothing,
    candidate: Option<(u64, Candidate)>,
    pub(crate) movement: Dispatcher<D>,
    last_ui: Option<Instant>,
    last_stop: Option<StopDiagnostic>,
}

struct RealTracking {
    profiles: HashSet<u32>,
    trial: bool,
    armed_frame: u64,
    crop: [u32; 2],
}

impl Default for AutomationController<DryRunDriver> {
    fn default() -> Self { Self::with_driver(DryRunDriver) }
}

impl<D: KmBoxDriver> AutomationController<D> {
    pub(crate) fn with_driver(driver: D) -> Self {
        Self { bindings: Vec::new(), state: AutomationState::Idle, auto_preview_pending: true, real: None,
            trial_acknowledged: false,
            reason: "等待实时状态", active: None,
            buttons: 0, report_at: None, physical_report: None, confirmation: None, confirmation_used: false,
            link_at: 0, source: 0, source_name: String::new(), device: String::new(), model: 0, stream_live: false, video_at: None,
            model_ready: false, frame: None, lock: TargetLock::default(), smoothing: Smoothing::default(),
            candidate: None, movement: Dispatcher::new(driver), last_ui: None, last_stop: None }
    }
    pub fn validate_profiles(profiles: &[Profile]) -> Result<(), String> {
        if profiles.len() > 31 {
            return Err("最多保存 31 个启用方案".into());
        }
        let mut masks = HashSet::new();
        let mut ids = HashSet::new();
        for profile in profiles {
            profile.validate()?;
            if !masks.insert(profile.mask) {
                return Err(format!("热键冲突：{} 与其他启用方案使用相同按键", profile.name));
            }
            if !ids.insert(profile.id) {
                return Err("热键方案编号重复".into());
            }
        }
        Ok(())
    }

    pub fn configure(&mut self, profiles: Vec<Profile>) -> Result<Value, String> {
        Self::validate_profiles(&profiles)?;
        self.leave_real();
        self.bindings = profiles.into_iter().map(|profile| Binding { profile, released: false, down: false, latched: false }).collect();
        self.reset_activation();
        (self.state, self.reason) = self.readiness();
        Ok(self.snapshot())
    }

    fn armed(&self) -> bool {
        matches!(self.state, AutomationState::Armed | AutomationState::Running)
    }

    fn reset_activation(&mut self) {
        for binding in &mut self.bindings {
            binding.reset(self.buttons);
        }
        self.active = None;
        self.confirmation_used = false;
        self.clear_target();
    }

    fn clear_target(&mut self) {
        self.lock = TargetLock::default();
        self.smoothing.reset();
        self.clear_candidate();
    }

    fn clear_candidate(&mut self) {
        self.candidate = None;
        self.movement.clear();
    }

    pub fn disarm(&mut self, reason: &'static str) -> Value {
        self.auto_preview_pending = false;
        self.stop(AutomationState::Stopped, reason)
    }

    fn stop(&mut self, state: AutomationState, reason: &'static str) -> Value {
        if state == AutomationState::Stopped || (state == AutomationState::Fault && self.state != AutomationState::Fault) {
            let at = now_ms();
            let inference_frame_at = self.frame.as_ref().map(|frame| frame.frame_at);
            self.last_stop = Some(StopDiagnostic {
                id: self.last_stop.as_ref().map_or(1, |stop| stop.id + 1), at_ms: at, reason,
                previous_state: self.state, mode: if self.real.is_some() { "real" } else { "dry_run" },
                video_frame_at_ms: self.video_at, video_age_ms: self.video_at.and_then(|time| at.checked_sub(time)),
                inference_frame_at_ms: inference_frame_at, inference_age_ms: inference_frame_at.and_then(|time| at.checked_sub(time)),
                age_limit_ms: FRAME_TIMEOUT_MS, frame_sequence: self.frame.as_ref().map(|frame| frame.sequence),
                track_id: self.candidate.as_ref().map(|(_, candidate)| candidate.track_id),
                button_age_ms: self.report_at.map(|time| time.elapsed().as_millis() as u64),
                cleared_pending: self.movement.has_pending(),
            });
        }
        self.leave_real();
        self.reset_activation();
        self.state = state;
        self.reason = reason;
        self.snapshot()
    }

    fn leave_real(&mut self) {
        if self.real.take().is_some() { self.movement.driver.stop_live(); }
    }

    fn reset_input(&mut self, reason: &'static str) -> Value {
        let state = if self.armed() { AutomationState::Stopped } else { AutomationState::Idle };
        self.stop(state, reason)
    }

    pub fn request_model(&mut self, model: u32) -> Value {
        self.model = model;
        self.model_ready = false;
        self.frame = None;
        self.reset_input("正在切换推理引擎")
    }

    pub fn request_stream(&mut self, source: u32) -> Value {
        self.source_name.clear();
        self.source = source;
        self.stream_live = false;
        self.video_at = None;
        self.frame = None;
        self.movement.reset_stream();
        self.reset_input("NDI 连接已更改")
    }

    pub fn request_named_stream(&mut self, source: u32, name: String) -> Value {
        let update = self.request_stream(source);
        self.source_name = name;
        update
    }

    pub fn restart(&mut self) -> Value {
        self.stream_live = false;
        self.video_at = None;
        self.model_ready = false;
        self.frame = None;
        self.movement.reset_stream();
        self.reset_input("推理进程已重置")
    }

    fn blocker(&self) -> Option<&'static str> {
        if self.bindings.is_empty() { return Some("请启用一个热键方案"); }
        if !self.stream_live { return Some("等待 NDI 视频"); }
        if self.video_at.is_none_or(|timestamp| !frame_fresh(timestamp)) {
            return Some("NDI 最新帧超时");
        }
        if !self.model_ready { return Some("等待模型就绪"); }
        if self.frame.as_ref().is_none_or(|frame| !frame_fresh(frame.frame_at)) {
            return Some("推理结果已过期");
        }
        if self.real.is_some() && !recent(self.link_at, LINK_CONFIRMATION_TIMEOUT_MS) {
            return Some("KMBox 连接应答已过期");
        }
        if self.confirmation_used && !self.confirmed_hold() {
            return Some("设备 A 持键确认或 KMBox 连接应答已过期");
        }
        if self.report_at.is_none_or(|at| at.elapsed() > BUTTON_TIMEOUT) && !self.allows_report_gap() {
            return Some("等待新鲜的 KMBox 物理按键报文");
        }
        None
    }

    pub fn arm(&mut self) -> Result<Value, String> {
        if self.armed() {
            return Ok(self.snapshot());
        }
        if let Some(reason) = self.blocker() { return Err(reason.into()); }
        self.reset_activation();
        self.auto_preview_pending = false;
        self.state = AutomationState::Armed;
        self.reason = "等待物理热键";
        Ok(self.snapshot())
    }

    pub fn arm_real(&mut self, profile_id: u32, full: bool) -> Result<Value, String> {
        if self.real.is_some() { return Err("请先停止当前的真实跟随".into()); }
        if full && !self.trial_acknowledged { return Err("请先在本次运行中完成单目标低速验收".into()); }
        if let Some(reason) = self.blocker() { return Err(reason.into()); }
        if !recent(self.link_at, LINK_CONFIRMATION_TIMEOUT_MS) || self.physical_report.is_none()
            || self.report_at.is_none_or(|at| at.elapsed() > BUTTON_TIMEOUT)
        {
            return Err("请先确认 KMBox 监控在线，并刷新物理按键报文".into());
        }
        if self.buttons != 0 { return Err("先松开鼠标按键，再武装真实跟随".into()); }
        let frame = self.frame.as_ref().ok_or("等待推理结果")?;
        let bindings: Vec<_> = self.bindings.iter().filter(|binding| !binding.profile.toggle
            && (full || binding.profile.id == profile_id)).collect();
        if bindings.is_empty() { return Err("请选择已启用的按住式热键方案".into()); }
        if !full {
            let profile = &bindings[0].profile;
            if profile.mask != 3 || profile.classes.as_ref().is_none_or(|ids| ids.len() != 1) {
                return Err("单目标验收需要左键 + 右键组合，并选择一个类别".into());
            }
        }
        for binding in &bindings {
            let profile = &binding.profile;
            let valid = profile.calibration_site.as_ref().is_some_and(|site| {
                site.source == self.source_name && !site.source.is_empty() && site.device == self.device && !site.device.is_empty()
                    && site.crop == [frame.width, frame.height]
                    && site.crop.into_iter().zip(site.desktop).all(|(crop, desktop)| crop > 0 && crop <= desktop)
                    && site.observed.into_iter().zip([(profile.gain_x, profile.invert_x), (profile.gain_y, profile.invert_y)])
                        .all(|(measured, (gain, invert))| measured.is_finite()
                            && measured.is_sign_negative() == invert && (10.0 / measured.abs() - gain).abs() < 0.000001)
            });
            if !valid { return Err(format!("{}：请校准当前 NDI 源和裁剪尺寸", profile.name)); }
        }
        let range = if full { f64::INFINITY } else { 64.0 };
        if !bindings.iter().any(|binding| frame.select_with_range(&binding.profile, &mut TargetLock::default(), &mut Smoothing::default(),
            range, binding.profile.tracking_gain).is_some()) {
            return Err("当前画面没有该方案可锁定的目标".into());
        }
        let real = RealTracking { profiles: bindings.iter().map(|binding| binding.profile.id).collect(),
            trial: !full, armed_frame: frame.sequence, crop: [frame.width, frame.height] };
        self.movement.driver.start_live()?;
        self.auto_preview_pending = false;
        self.reset_activation();
        self.real = Some(real);
        self.state = AutomationState::Armed;
        self.reason = "真实跟随已武装，等待重新按下热键";
        Ok(self.snapshot())
    }

    pub fn monitor_changed(&mut self) -> Value {
        self.report_at = None;
        self.physical_report = None;
        self.link_at = 0;
        self.buttons = 0;
        self.reset_input("KMBox 监控正在切换")
    }

    pub fn request_device(&mut self, device: &str) -> Value {
        let device = device.chars().filter(|character| character.is_ascii_hexdigit()).collect::<String>().to_ascii_uppercase();
        if self.device != device { self.trial_acknowledged = false; }
        self.device = device;
        self.monitor_changed()
    }

    pub fn monitor_failed(&mut self) -> Value {
        self.link_at = 0;
        self.buttons = 0;
        let update = self.stop(AutomationState::Fault, "KMBox 监控已断开或报文过期");
        self.report_at = None;
        update
    }

    fn held_mask(&self) -> Option<u8> {
        self.active.and_then(|index| self.bindings.get(index)).filter(|binding| !binding.profile.toggle)
            .map(|binding| binding.profile.mask)
    }

    fn confirmed_hold(&self) -> bool {
        self.confirmation_used && recent(self.link_at, LINK_CONFIRMATION_TIMEOUT_MS) &&
            self.confirmation.zip(self.held_mask()).is_some_and(|((buttons, at), mask)| {
                recent(at, CONFIRMATION_TIMEOUT_MS) && buttons & mask == mask
            })
    }

    fn event_preview(&self) -> bool {
        self.real.is_none() && self.active.is_some_and(|i| self.bindings[i].profile.event_preview && !self.bindings[i].profile.toggle)
    }

    fn has_event_preview(&self) -> bool {
        self.bindings.iter().any(|binding| binding.profile.event_preview && !binding.profile.toggle)
    }

    fn allows_report_gap(&self) -> bool {
        self.real.is_none() && (self.confirmed_hold() || (recent(self.link_at, LINK_CONFIRMATION_TIMEOUT_MS)
            && (self.event_preview() || (self.active.is_none() && self.has_event_preview()))))
    }

    pub fn confirmation_changed(&mut self) -> Value {
        self.confirmation = None;
        self.disarm("设备 A 按键确认已更改，请手动启用 Dry Run")
    }

    pub fn confirm_buttons(&mut self, event: &Value) -> Option<Value> {
        if self.real.is_some() || self.event_preview() { return None; }
        let at = event["at_ms"].as_u64().unwrap_or_default();
        if at == 0 {
            self.confirmation = None;
            return self.tick();
        }
        let mask = event["mask"].as_u64().filter(|mask| *mask < 32)? as u8;
        if !recent(at, CONFIRMATION_TIMEOUT_MS) || self.confirmation.is_some_and(|(_, last)| at <= last) {
            return None;
        }
        if self.confirmation_used && !self.confirmed_hold() {
            return Some(self.stop(AutomationState::Fault, "设备 A 持键确认或 KMBox 连接应答已过期"));
        }
        self.confirmation = Some((mask, at));
        let held = self.held_mask()?;
        if mask & held == held && recent(self.link_at, LINK_CONFIRMATION_TIMEOUT_MS)
            && self.report_at.is_some_and(|report| report.elapsed() <= BUTTON_TIMEOUT)
        {
            let newly_confirmed = !self.confirmation_used;
            self.confirmation_used = true;
            if newly_confirmed {
                return Some(self.snapshot());
            }
        } else if self.confirmation_used && mask & held != held {
            self.reset_activation();
            self.follow();
            self.reason = "设备 A 已确认热键释放，已清除候选位移";
            return Some(self.snapshot());
        }
        None
    }

    pub fn buttons(&mut self, event: &Value) -> Option<Value> {
        let buttons = event["buttons"].as_u64().filter(|mask| *mask <= 31);
        let age = event["age_ms"].as_u64().and_then(|reported_age| {
            match event["report_at_ms"].as_u64() {
                Some(timestamp) => now_ms().checked_sub(timestamp).map(|age| age.max(reported_age)),
                None => Some(reported_age),
            }
        }).map(Duration::from_millis);
        if event["connected"] != true || event["monitoring"] != true || event["halted"] == true {
            return if self.report_at.is_some() || self.armed() { Some(self.monitor_failed()) } else { self.refresh_readiness() };
        }
        self.link_at = self.link_at.max(event["link_at_ms"].as_u64().filter(|at| *at <= now_ms()).unwrap_or(0));
        if buttons.is_none() || age.is_none_or(|age| age > BUTTON_TIMEOUT) {
            return if self.allows_report_gap() { self.refresh_readiness() }
                else if self.report_at.is_some() || self.armed() { Some(self.monitor_failed()) }
                else { self.refresh_readiness() };
        }
        let expired = self.armed() && self.report_at.is_some_and(|at| at.elapsed() > BUTTON_TIMEOUT) && !self.allows_report_gap();
        let report = event["reports"].as_u64().filter(|count| *count != 0).zip(event["report_at_ms"].as_u64());
        if let Some((count, stamp)) = report {
            if self.physical_report.is_some_and(|(last, at)| count <= last || stamp < at) {
                return None;
            }
            self.physical_report = Some((count, stamp));
        }
        self.report_at = Instant::now().checked_sub(age.unwrap());
        self.buttons = buttons.unwrap() as u8;
        if expired {
            return Some(self.stop(AutomationState::Fault, "KMBox 物理按键报文已过期"));
        }
        if !self.armed() {
            return self.refresh_readiness();
        }
        if let Some(reason) = self.blocker() {
            return Some(self.stop(AutomationState::Fault, reason));
        }
        for binding in &mut self.bindings {
            binding.update(self.buttons);
        }
        if self.real.is_some() && self.active.is_some_and(|index| !self.bindings[index].down) {
            return Some(self.stop(AutomationState::Stopped, "真实跟随热键已释放，请重新武装"));
        }
        if self.active.is_some_and(|index| {
            let previous = &self.bindings[index];
            previous.profile.mask & 3 == 3 && !previous.down
        }) {
            // A held single key must never inherit control when the chord releases.
            self.reset_activation();
            self.follow();
            self.reason = "组合键释放，已清除候选位移";
            return Some(self.snapshot());
        }
        let next = self.bindings.iter().enumerate().filter(|(_, binding)| binding.active())
            .filter(|(_, binding)| self.real.as_ref().is_none_or(|real| real.profiles.contains(&binding.profile.id)))
            .max_by_key(|(index, binding)| (binding.profile.mask.count_ones(), std::cmp::Reverse(*index)))
            .map(|(index, _)| index);
        if self.active == next {
            return None;
        }
        if self.real.is_some() && self.active.is_some() {
            return Some(self.stop(AutomationState::Stopped, "热键方案已变化，请重新武装"));
        }
        self.clear_target();
        self.active = next;
        self.confirmation_used = false;
        if let Some(real) = &mut self.real {
            self.lock.waiting = true;
            if let Some(frame) = &self.frame { real.armed_frame = frame.sequence; }
        }
        self.follow();
        Some(self.snapshot())
    }

    fn readiness(&mut self) -> (AutomationState, &'static str) {
        if let Some(reason) = self.blocker() {
            return (AutomationState::Idle, reason);
        }
        if self.auto_preview_pending && self.has_event_preview() && recent(self.link_at, LINK_CONFIRMATION_TIMEOUT_MS) {
            self.auto_preview_pending = false;
            self.reset_activation();
            return (AutomationState::Armed, "已自动启用，等待物理热键");
        }
        (AutomationState::Ready, "已就绪，等待手动启用 Dry Run")
    }

    fn refresh_readiness(&mut self) -> Option<Value> {
        if !matches!(self.state, AutomationState::Idle | AutomationState::Ready) {
            return None;
        }
        let (state, reason) = self.readiness();
        if self.state == state && self.reason == reason {
            return None;
        }
        self.state = state;
        self.reason = reason;
        Some(self.snapshot())
    }

    pub fn tick(&mut self) -> Option<Value> {
        self.tick_at(Instant::now())
    }

    pub(crate) fn tick_at(&mut self, now: Instant) -> Option<Value> {
        if self.armed() {
            if let Some(reason) = self.blocker() {
                return Some(self.stop(AutomationState::Fault, reason));
            }
            if self.state == AutomationState::Running {
                if let Err(reason) = self.movement.dispatch(now) {
                    return Some(self.stop(AutomationState::Fault, reason));
                }
            }
        }
        self.refresh_readiness()
    }

    fn follow(&mut self) {
        if self.real.as_ref().zip(self.frame.as_ref()).is_some_and(|(real, frame)| frame.sequence <= real.armed_frame) {
            return;
        }
        let previously_locked = self.candidate.is_some();
        let selected = self.active.and_then(|index| self.bindings.get(index));
        let resolution = selected.zip(self.frame.as_ref()).and_then(|(binding, frame)| {
            let intent = if let Some(real) = &self.real {
                let radius = if real.trial { 64.0 } else { f64::INFINITY };
                frame.select_with_range(&binding.profile, &mut self.lock, &mut self.smoothing, radius, binding.profile.tracking_gain)
            } else {
                frame.select(&binding.profile, &mut self.lock, &mut self.smoothing)
            };
            intent.map(|intent| (frame.frame_at, intent))
        });
        if let Some((timestamp, intent)) = resolution {
            let held = selected.filter(|binding| !binding.profile.toggle).and_then(|binding| {
                self.physical_report.map(|(report, at_ms)| HeldInput { mask: binding.profile.mask, report, at_ms })
            });
            if let Some((real, binding)) = self.real.as_ref().zip(selected) {
                let speed = if real.trial { binding.profile.max_speed.min(120) } else { binding.profile.max_speed };
                self.movement.offer_live(timestamp, intent.sequence, intent.relative, Instant::now(), held,
                    binding.profile.motion.speed(intent.offset[0].hypot(intent.offset[1]), speed),
                    if real.trial { binding.profile.acceleration.min(1000) } else { binding.profile.acceleration },
                    (intent.lock_id, intent.class));
            } else if !self.event_preview() {
                self.movement.offer(timestamp, intent.sequence, intent.relative, Instant::now(), held);
            }
            self.candidate = Some((timestamp, intent));
            self.state = AutomationState::Running;
            self.reason = if self.real.is_some() { "真实跟随" } else { "Dry Run" };
        } else {
            if self.real.is_some() && previously_locked {
                self.stop(AutomationState::Fault, "锁定目标已丢失，请重新武装");
                return;
            }
            self.reason = if self.active.is_none() { "等待物理热键" }
                else if self.lock.waiting { "锁定目标已离开，等待确认新目标" }
                else { NO_TARGET };
            self.clear_candidate();
            self.state = AutomationState::Armed;
        }
    }

    pub fn worker(&mut self, event: &Value) -> Option<Value> {
        match event["kind"].as_str()? {
            "stream_state" if event["source"] == self.source => {
                self.stream_live = matches!(event["state"].as_str(), Some("live" | "resumed"));
                if !self.stream_live {
                    let update = self.stop(AutomationState::Fault, "NDI 断流");
                    self.video_at = None;
                    self.frame = None;
                    return Some(update);
                }
                self.video_at = event["frame_at"].as_u64().filter(|timestamp| *timestamp != 0);
            }
            "ndi_health" if self.stream_live && event["source"] == self.source => {
                if let Some(timestamp) = event["frame_at"].as_u64().filter(|timestamp| *timestamp <= now_ms()) {
                    self.video_at = self.video_at.max(Some(timestamp));
                }
            }
            "stream" if event["source"] == self.source && event.get("error").is_some() => {
                self.stream_live = false;
                let update = self.stop(AutomationState::Fault, "NDI 接收故障");
                self.video_at = None;
                self.frame = None;
                return Some(update);
            }
            "model" if event["model"] == self.model => {
                self.model_ready = event.get("error").is_none() && event["provider"].is_string();
                if !self.model_ready {
                    let update = self.stop(AutomationState::Fault, "模型故障");
                    self.frame = None;
                    return Some(update);
                }
            }
            "detections" => {
                let frame: Frame = serde_json::from_value(event.clone()).ok()?;
                if !self.model_ready || !self.stream_live || frame.model != self.model || frame.source != self.source
                    || !frame_fresh(frame.frame_at) || self.frame.as_ref().is_some_and(|last| last.sequence >= frame.sequence)
                {
                    return None;
                }
                if self.real.as_ref().is_some_and(|real| real.crop != [frame.width, frame.height]) {
                    self.frame = Some(frame);
                    return Some(self.stop(AutomationState::Fault, "NDI 裁剪尺寸已变化，请重新校准"));
                }
                self.frame = Some(frame);
                if self.armed() {
                    if let Some(reason) = self.blocker() {
                        return Some(self.stop(AutomationState::Fault, reason));
                    }
                    if self.active.is_some() {
                        let previous = self.state;
                        self.follow();
                        if previous != self.state || self.last_ui.is_none_or(|at| at.elapsed() >= Duration::from_millis(33)) {
                            self.last_ui = Some(Instant::now());
                            return Some(self.snapshot());
                        }
                    }
                }
            }
            "worker_exit" => {
                self.stream_live = false;
                let update = self.stop(AutomationState::Fault, "推理进程已退出");
                self.video_at = None;
                self.model_ready = false;
                self.frame = None;
                return Some(update);
            }
            _ => {}
        }
        self.refresh_readiness()
    }

    fn snapshot(&self) -> Value {
        let candidate = self.current_candidate();
        let active = self.active.and_then(|index| self.bindings.get(index));
        json!({"kind":"dry_run", "state":self.state, "armed":self.armed(), "active":active.is_some(),
               "profile_id":active.map(|binding| binding.profile.id), "profile_name":active.map(|binding| &binding.profile.name),
               "candidate":candidate, "reason":self.reason,
               "mode":if self.real.is_some() { "real" } else { "dry_run" },
               "real_control_available":self.real.is_some(),
               "scheduled":self.movement.submitted, "expired":self.movement.expired,
               "queue_ms":self.movement.last_queue_ms,
               "last_stop":self.last_stop,
               "hold_confirmed":self.confirmed_hold(),
               "event_hold_age_ms":self.report_at.filter(|at| self.event_preview() && at.elapsed() > BUTTON_TIMEOUT)
                   .map(|at| at.elapsed().as_millis() as u64)})
    }

    fn current_candidate(&self) -> Option<&Candidate> {
        self.candidate.as_ref().filter(|(timestamp, _)| frame_fresh(*timestamp)).map(|(_, intent)| intent)
    }

    pub(crate) fn target_update(&self) -> Option<(u64, u64)> {
        self.current_candidate().map(|candidate| (candidate.computed_at_ms, candidate.sequence))
    }

    pub(crate) fn target_sample(&self) -> Option<(u64, u64, f64)> {
        self.current_candidate().map(|candidate| (candidate.sequence, candidate.latency_ms, candidate.decision_ms))
    }

    pub(crate) fn target_track(&self) -> Option<u64> {
        self.current_candidate().map(|candidate| candidate.track_id)
    }

    pub(crate) fn live_fault(&mut self, reason: &'static str) -> Option<Value> {
        self.real.is_some().then(|| self.stop(AutomationState::Fault, reason))
    }

    pub(crate) fn record_trial_ack(&mut self) {
        if self.real.as_ref().is_some_and(|real| real.trial) { self.trial_acknowledged = true; }
    }

    pub(crate) fn source_revision(&self) -> u32 { self.source }
}

pub(crate) fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

pub(crate) fn frame_fresh(timestamp: u64) -> bool {
    now_ms().checked_sub(timestamp).is_some_and(|age| age <= FRAME_TIMEOUT_MS)
}

#[derive(Deserialize)]
struct RecordedFrame {
    at_ms: u64,
    boxes: Vec<[f64; 6]>,
}

pub(crate) fn replay_tracking(input: &Value) -> Result<Value, String> {
    let frames: Vec<RecordedFrame> = serde_json::from_value(input["frames"].clone()).map_err(|e| e.to_string())?;
    let [width, height]: [u32; 2] = serde_json::from_value(input["dimensions"].clone()).map_err(|e| e.to_string())?;
    if width == 0 || height == 0 || frames.is_empty() || frames.windows(2).any(|pair| pair[0].at_ms >= pair[1].at_ms) {
        return Err("回放需要有效尺寸和严格递增的帧时间".into());
    }
    let mut comparison = serde_json::Map::new();
    for algorithm in ["p", "pd"] {
        let mut profile = input["profile"].clone();
        profile["motion"]["algorithm"] = json!(algorithm);
        let profile: Profile = serde_json::from_value(profile).map_err(|e| e.to_string())?;
        profile.validate()?;
        let (mut tracker, mut smoothing) = (TargetLock::default(), Smoothing::default());
        let mut decisions = Vec::new();
        for (index, frame) in frames.iter().enumerate() {
            let frame = Frame { source: 0, model: 0, frame_at: frame.at_ms, sequence: index as u64 + 1,
                width, height, boxes: frame.boxes.clone() };
            let result = frame.select(&profile, &mut tracker, &mut smoothing).map(|candidate| {
                json!({"track":candidate.track_id, "lock":candidate.lock_id, "center":candidate.center,
                    "anchor":candidate.anchor, "class":candidate.class,"delta":candidate.relative,
                    "offset":candidate.offset, "velocity_px_s":candidate.target_velocity_px_s, "radius_px":candidate.range_px,
                    "reason":candidate.selection_reason})
            });
            decisions.push(result);
        }
        comparison.insert(algorithm.into(), json!(decisions));
    }
    Ok(json!({"frames":frames.len(), "comparisons":comparison}))
}

fn recent(timestamp: u64, window: u64) -> bool {
    timestamp != 0 && now_ms().checked_sub(timestamp).is_some_and(|age| age <= window)
}

#[cfg(test)]
mod tracking_tests {
    use super::*;

    fn detection(x: f64, category: f64) -> [f64; 6] {
        [x - 22.0, 298.0, x + 22.0, 342.0, 0.9, category]
    }

    #[test]
    fn holds_track_through_crossing_and_recovers_after_one_missing_frame() {
        let p = Profile { enabled_classes: vec![0], lost_wait_ms: 120, ..Profile::default() };
        let mut lock = TargetLock::default();
        let mut smooth = Smoothing::default();
        let mut select = |seq, boxes| {
            Frame { source: 1, model: 1, width: 640, height: 640, sequence: seq,
                frame_at: seq * 16, boxes }.select(&p, &mut lock, &mut smooth)
        };
        let start = select(1, vec![detection(280.0, 0.0), detection(375.0, 0.0)]).unwrap();
        for (seq, a, b) in [(2,294.0,358.0), (3,308.0,341.0), (4,322.0,324.0), (5,336.0,307.0)] {
            let next = select(seq, vec![detection(b, 0.0), detection(a, 0.0)]).unwrap();
            assert_eq!(next.track_id, start.track_id);
            assert_eq!(next.center[0], a);
        }
        assert!(select(6, vec![detection(290.0, 0.0)]).is_none());
        let restored = select(7, vec![detection(273.0, 0.0), detection(364.0, 0.0)]).unwrap();
        assert_eq!(restored.track_id, start.track_id);
        assert_eq!(restored.lock_id, start.lock_id);
        assert_eq!(restored.selection_reason, "恢复同一 Track ID");
    }

    #[test]
    fn waits_after_loss_and_applies_independent_class_anchor() {
        let profile = serde_json::from_value(json!({"enabled_classes":[0,1], "lost_wait_ms":40,
            "per_class":{"1":{"threshold":0.85,"priority":2,"anchor":[25,-25]}}})).unwrap();
        let mut lock = TargetLock::default();
        let mut smooth = Smoothing::default();
        let mut select = |sequence, frame_at, boxes| {
            Frame { source:1, model:1, width:640, height:640, sequence, frame_at, boxes }
                .select(&profile, &mut lock, &mut smooth)
        };
        let first = select(1, 16, vec![detection(320.0, 0.0), detection(354.0, 1.0)]).unwrap();
        assert_eq!(first.class, 1);
        assert_eq!(first.anchor, [365.0,309.0]);
        assert!(select(2, 32, vec![detection(320.0, 0.0)]).is_none());
        assert!(select(3, 64, vec![detection(320.0, 0.0)]).is_none());
        let replacement = select(4, 80, vec![detection(320.0, 0.0)]).unwrap();
        assert_ne!(replacement.track_id, first.track_id);
        assert_eq!(replacement.selection_reason, "新目标已连续确认");
    }

    #[test]
    fn dynamic_radius_expands_only_for_the_locked_target() {
        let p = Profile { enabled_classes: vec![0], range_percent: 50.0, dynamic_range_percent: Some(5.0),
            ..Profile::default() };
        let mut lock = TargetLock::default();
        let mut smooth = Smoothing::default();
        let mut select = |seq, x| {
            Frame { source:1, model:1, width:640, height:640, sequence:seq, frame_at:seq * 16,
                boxes:vec![detection(x, 0.0)] }.select(&p, &mut lock, &mut smooth)
        };
        assert!(select(1, 342.0).is_none());
        let start = select(2, 324.0).unwrap();
        assert_eq!(start.range_px, 16.0);
        select(3, 332.0).unwrap();
        let expanded = select(4, 348.0).unwrap();
        assert_eq!(expanded.track_id, start.track_id);
        assert!(expanded.range_px > 28.0 && expanded.range_px <= 160.0);
    }
}
