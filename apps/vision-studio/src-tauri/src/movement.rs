use crate::control::{frame_fresh, now_ms, FRAME_TIMEOUT_MS};
use serde::{Deserialize, Serialize};
use std::{io::{self, Write}, process::ChildStdin, sync::{Arc, Mutex}, time::{Duration, Instant}};

pub const KMBOX_AXIS_LIMIT: i32 = 24;
const COMMAND_TTL: Duration = Duration::from_millis(45);
pub const DISPATCH_INTERVAL: Duration = Duration::from_millis(17);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct RelativeMove {
    pub dx: i32,
    pub dy: i32,
}

#[derive(Clone, Copy)]
pub struct HeldInput {
    pub mask: u8,
    pub report: u64,
    pub at_ms: u64,
}

#[derive(Clone, Copy)]
pub struct MoveContext {
    pub id: u64,
    pub expires_at_ms: u64,
    pub held: Option<HeldInput>,
    target: Option<(u64, u16, u64)>,
    queue_ms: f64,
}

pub trait KmBoxDriver {
    fn submit_relative(&mut self, movement: RelativeMove, context: MoveContext) -> Result<(), &'static str>;
    fn ready(&self) -> bool { true }
    fn can_arm_live(&self) -> bool { false }
    fn start_live(&mut self) -> Result<(), &'static str> { Err("真实控制接口未连接") }
    fn take_over(&mut self, _: HeldInput) -> Result<(), &'static str> { Err("真实控制接口未连接") }
    fn axes_taken(&self) -> bool { false }
    fn stop_live(&mut self) {}
}

#[derive(Default)]
pub struct DryRunDriver;

impl KmBoxDriver for DryRunDriver {
    fn submit_relative(&mut self, _: RelativeMove, _: MoveContext) -> Result<(), &'static str> {
        Ok(())
    }
}

pub(crate) struct KmBoxPipe<W: Write>(pub W);

impl<W: Write> KmBoxDriver for KmBoxPipe<W> {
    fn submit_relative(&mut self, movement: RelativeMove, context: MoveContext) -> Result<(), &'static str> {
        let held = context.held.ok_or("KMBox 位移需要有效的按住式热键报文")?;
        #[derive(Serialize)]
        struct Request {
            cmd: &'static str,
            dx: i32,
            dy: i32,
            id: u64,
            expires_at_ms: u64,
            hold_mask: u8,
            report: u64,
            report_at_ms: u64,
        }
        let request = Request { cmd: "relative_move", dx: movement.dx, dy: movement.dy,
            id: context.id, expires_at_ms: context.expires_at_ms, hold_mask: held.mask,
            report: held.report, report_at_ms: held.at_ms };
        serde_json::to_writer(&mut self.0, &request).map_err(|_| "KMBox 命令传输失败")?;
        self.0.write_all(b"\n").and_then(|_| self.0.flush()).map_err(|_| "KMBox 命令传输失败")
    }
}

#[derive(Clone)]
pub(crate) struct MonitorWriter(pub Arc<Mutex<ChildStdin>>);

impl MonitorWriter {
    pub fn message(&self, value: &impl Serialize) -> io::Result<()> {
        let mut bytes = serde_json::to_vec(value)?;
        bytes.push(b'\n');
        self.write_message(&bytes)
    }

    fn write_message(&self, bytes: &[u8]) -> io::Result<()> {
        let mut pipe = self.0.lock().map_err(|_| io::Error::other("KMBox pipe lock"))?;
        pipe.write_all(bytes)?;
        pipe.flush()
    }
}

#[derive(Default)]
pub(crate) struct RuntimeDriver {
    pub pipe: Option<MonitorWriter>,
    permission: LivePermission,
    restored: bool,
    box_busy: bool,
    recovery_unconfirmed: bool,
    takeover_id: u64,
    outstanding: Option<(MoveContext, Instant)>,
}

#[derive(Default)]
enum LivePermission {
    #[default]
    Stopped,
    Armed,
    Masking(Instant),
    Active,
}

impl RuntimeDriver {
    pub fn observe_axes(&mut self, status: Option<&str>, busy: bool) {
        self.restored = status == Some("restored");
        self.box_busy = busy;
        match status {
            Some("restored") => self.recovery_unconfirmed = false,
            Some("masking" | "masked" | "restoring" | "recovery_pending") => self.recovery_unconfirmed = true,
            _ => {}
        }
    }

    pub fn takeover_result(&mut self, id: u64, acknowledged: bool) -> Option<bool> {
        if id != self.takeover_id || !matches!(self.permission, LivePermission::Masking(_)) { return None; }
        if acknowledged { self.permission = LivePermission::Active; }
        Some(acknowledged)
    }

    pub fn axes_need_recovery(&self) -> bool {
        matches!(self.permission, LivePermission::Masking(_) | LivePermission::Active) || self.recovery_unconfirmed
    }

    pub fn result(&mut self, id: u64) -> Option<(Option<(u64, u16, u64)>, Duration, f64)> {
        let (pending, started) = self.outstanding?;
        if id != pending.id { return None; }
        self.outstanding = None;
        Some((pending.target, started.elapsed(), pending.queue_ms))
    }

    pub fn timed_out(&self) -> bool {
        self.outstanding.is_some_and(|(_, at)| at.elapsed() > Duration::from_millis(300)) ||
            matches!(self.permission, LivePermission::Masking(at) if at.elapsed() > Duration::from_millis(600))
    }

    pub fn detach(&mut self) {
        self.reset_connection();
        self.pipe = None;
    }

    pub fn reset_connection(&mut self) {
        self.permission = LivePermission::Stopped;
        self.restored = false;
        self.outstanding = None;
    }
}

impl KmBoxDriver for RuntimeDriver {
    fn submit_relative(&mut self, movement: RelativeMove, context: MoveContext) -> Result<(), &'static str> {
        if context.target.is_none() { return Ok(()); }
        if !self.axes_taken() { return Err("KMBox 尚未应答屏蔽物理 X/Y"); }
        let pipe = self.pipe.as_ref().ok_or("KMBox 控制进程未连接")?;
        let mut request = KmBoxPipe(Vec::new());
        request.submit_relative(movement, context)?;
        pipe.write_message(&request.0).map_err(|_| "KMBox 命令传输失败")?;
        self.outstanding = Some((context, Instant::now()));
        Ok(())
    }

    fn ready(&self) -> bool { self.outstanding.is_none() }

    fn can_arm_live(&self) -> bool { self.restored && !self.box_busy && self.ready() }

    fn start_live(&mut self) -> Result<(), &'static str> {
        if self.outstanding.is_some() { return Err("上一条 KMBox 命令尚未完成"); }
        if !self.restored { return Err("请先确认 KMBox 已解除物理 X/Y 屏蔽"); }
        self.pipe.as_ref().ok_or("KMBox 控制进程未连接")?
            .message(&serde_json::json!({"cmd":"auto_enable"})).map_err(|_| "KMBox 武装传输失败")?;
        self.permission = LivePermission::Armed;
        Ok(())
    }

    fn take_over(&mut self, held: HeldInput) -> Result<(), &'static str> {
        if !matches!(self.permission, LivePermission::Armed) { return Err("KMBox 未武装"); }
        self.takeover_id += 1;
        self.restored = false;
        self.recovery_unconfirmed = true;
        self.permission = LivePermission::Masking(Instant::now());
        self.pipe.as_ref().ok_or("KMBox 控制进程未连接")?.message(&serde_json::json!({
            "cmd":"auto_takeover", "id":self.takeover_id, "report":held.report, "report_at_ms":held.at_ms,
        })).map_err(|_| "KMBox 接管传输失败")?;
        Ok(())
    }

    fn axes_taken(&self) -> bool { matches!(self.permission, LivePermission::Active) }

    fn stop_live(&mut self) {
        if !matches!(self.permission, LivePermission::Stopped) {
            self.permission = LivePermission::Stopped;
            self.outstanding = None;
            if let Some(pipe) = &self.pipe {
                let _ = pipe.message(&serde_json::json!({"cmd":"auto_stop"}));
            }
        }
    }
}

#[derive(Clone, Copy)]
struct PendingMove {
    movement: RelativeMove,
    sequence: u64,
    frame_at: u64,
    created: Instant,
    expires_at_ms: u64,
    held: Option<HeldInput>,
    limits: Option<(u16, u16)>,
    target: Option<(u64, u16, u64)>,
}

pub struct Dispatcher<D> {
    pub(crate) driver: D,
    pending: Option<PendingMove>,
    last: Option<(Instant, u64)>,
    pub(crate) submitted: u64,
    pub(crate) expired: u64,
    pub(crate) last_queue_ms: Option<f64>,
    live_rate: [f64; 2],
    live_fraction: [f64; 2],
}

impl<D: KmBoxDriver> Dispatcher<D> {
    pub fn new(driver: D) -> Self {
        Self { driver, pending: None, last: None, submitted: 0, expired: 0,
            last_queue_ms: None, live_rate: [0.0; 2], live_fraction: [0.0; 2] }
    }

    pub fn offer(&mut self, frame_at: u64, sequence: u64, [dx, dy]: [i32; 2], now: Instant, held: Option<HeldInput>) {
        let within_limit = |value: i32| value.clamp(-KMBOX_AXIS_LIMIT, KMBOX_AXIS_LIMIT);
        self.pending = (dx != 0 || dy != 0).then_some(PendingMove { frame_at, sequence,
            movement: RelativeMove { dx: within_limit(dx), dy: within_limit(dy) }, created: now, held, limits: None, target: None,
            expires_at_ms: (now_ms() + COMMAND_TTL.as_millis() as u64).min(frame_at + FRAME_TIMEOUT_MS) });
    }

    pub fn offer_live(&mut self, frame_at: u64, sequence: u64, relative: [i32; 2], now: Instant,
        held: Option<HeldInput>, speed: u16, acceleration: u16, target: (u64, u16)) {
        self.offer(frame_at, sequence, relative, now, held);
        if let Some(command) = &mut self.pending {
            command.limits = Some((speed, acceleration));
            command.target = Some((target.0, target.1, frame_at));
        } else {
            self.live_rate = [0.0; 2];
            self.live_fraction = [0.0; 2];
        }
    }

    pub fn clear(&mut self) {
        self.pending = None;
        self.live_rate = [0.0; 2];
        self.live_fraction = [0.0; 2];
        self.last_queue_ms = None;
    }

    pub(crate) fn has_pending(&self) -> bool { self.pending.is_some() }

    pub fn reset_stream(&mut self) {
        self.clear();
        self.last = None;
    }

    pub fn dispatch(&mut self, now: Instant) -> Result<(), &'static str> {
        let Some(command) = self.pending else { return Ok(()); };
        if now.duration_since(command.created) > COMMAND_TTL || now_ms() >= command.expires_at_ms || !frame_fresh(command.frame_at) {
            self.clear();
            self.expired += 1;
            if command.limits.is_some() { return Err("待发送位移已超时，请重新武装"); }
        } else if let Some((last, sequence)) = self.last {
            if command.sequence <= sequence {
                self.clear();
            } else if now.duration_since(last) >= DISPATCH_INTERVAL {
                if self.driver.ready() { self.submit(command, now)?; }
            }
        } else if self.driver.ready() {
            self.submit(command, now)?;
        }
        Ok(())
    }

    fn submit(&mut self, command: PendingMove, now: Instant) -> Result<(), &'static str> {
        self.pending = None;
        let movement = self.limit_live(command);
        if movement.dx != 0 || movement.dy != 0 {
            let queue_ms = now.saturating_duration_since(command.created).as_secs_f64() * 1000.0;
            let context = MoveContext { id: self.submitted + 1, expires_at_ms: command.expires_at_ms,
                held: command.held, target: command.target, queue_ms };
            self.driver.submit_relative(movement, context)?;
            self.submitted += 1;
            self.last_queue_ms = Some(queue_ms);
        }
        self.last = Some((now, command.sequence));
        Ok(())
    }

    fn limit_live(&mut self, command: PendingMove) -> RelativeMove {
        let Some((speed, acceleration)) = command.limits else { return command.movement; };
        let interval = DISPATCH_INTERVAL.as_secs_f64();
        let speed = f64::from(speed);
        let acceleration_step = f64::from(acceleration) * interval;
        let desired = [command.movement.dx, command.movement.dy].map(|axis| f64::from(axis) / interval);
        let magnitude = desired[0].hypot(desired[1]);
        let scale = (speed / magnitude).min(1.0);
        let desired = desired.map(|value| value * scale);
        for (axis, target) in desired.into_iter().enumerate() {
            let previous = &mut self.live_rate[axis];
            if target == 0.0 || target.signum() != previous.signum() {
                *previous = 0.0;
                self.live_fraction[axis] = 0.0;
            }
            if target.abs() < previous.abs() { *previous = target; }
        }
        let increase = [desired[0] - self.live_rate[0], desired[1] - self.live_rate[1]];
        let ramp = (acceleration_step / increase[0].hypot(increase[1])).min(1.0);
        self.live_rate[0] += increase[0] * ramp;
        self.live_rate[1] += increase[1] * ramp;
        let step = self.live_rate.map(|rate| rate * interval);
        let limit = step[0].hypot(step[1]).min(speed * interval);
        let biased: [f64; 2] = std::array::from_fn(|axis| step[axis] + self.live_fraction[axis]);
        let distance = |[x, y]: [f64; 2]| (x - biased[0]).powi(2) + (y - biased[1]).powi(2);
        let mut chosen = step.map(f64::trunc);
        // Carry sub-unit motion between sends while keeping each integer step inside the current limit.
        for x in [step[0].floor(), step[0].ceil()] {
            for y in [step[1].floor(), step[1].ceil()] {
                if x.hypot(y) <= limit && distance([x, y]) < distance(chosen) {
                    chosen = [x, y];
                }
            }
        }
        self.live_fraction = std::array::from_fn(|axis| (biased[axis] - chosen[axis]).clamp(-1.0, 1.0));
        let [dx, dy] = chosen.map(|axis| axis as i32);
        RelativeMove { dx, dy }
    }
}

#[derive(Default)]
pub struct Smoothing {
    previous: Option<(u64, [f64; 2])>,
}

#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Algorithm {
    #[default]
    P,
    Pd,
}

#[derive(Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MotionPhase {
    Acquisition,
    Tracking,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(default)]
pub(crate) struct MotionConfig {
    algorithm: Algorithm,
    derivative_ms: [f64; 2],
    prediction_ms: f64,
    slow_radius_px: f64,
    adaptive: bool,
    acquisition_gain: f64,
    transition_px: f64,
}

impl Default for MotionConfig {
    fn default() -> Self {
        Self { algorithm: Algorithm::P, derivative_ms: [0.0; 2], prediction_ms: 0.0,
            slow_radius_px: 0.0, adaptive: false, acquisition_gain: 2.5, transition_px: 24.0 }
    }
}

impl MotionConfig {
    pub fn validate(&self) -> bool {
        self.derivative_ms.into_iter().all(|v| (0.0..=120.0).contains(&v))
            && (0.0..=80.0).contains(&self.prediction_ms)
            && (0.0..=160.0).contains(&self.slow_radius_px)
            && (1.0..=8.0).contains(&self.acquisition_gain)
            && (1.0..=160.0).contains(&self.transition_px)
    }

    pub fn phase(&self, previous: Option<MotionPhase>, distance: f64) -> MotionPhase {
        // A wider exit threshold keeps jitter around the boundary from switching modes.
        let threshold = if previous == Some(MotionPhase::Tracking) { self.transition_px * 1.5 }
            else { self.transition_px };
        if distance > threshold { MotionPhase::Acquisition } else { MotionPhase::Tracking }
    }

    pub fn for_acquisition(self, distance: f64) -> (Self, f64) {
        let boost = ((distance - self.transition_px) / self.transition_px).clamp(0.0, 1.0);
        (Self { algorithm: Algorithm::P, prediction_ms: 0.0, slow_radius_px: 0.0, adaptive: false, ..self },
            1.0 + (self.acquisition_gain - 1.0) * boost)
    }

    fn slowdown(&self, distance: f64) -> f64 {
        if self.slow_radius_px == 0.0 { 1.0 } else { (distance / self.slow_radius_px).clamp(0.25, 1.0) }
    }

    pub fn speed(&self, distance: f64, maximum: u16) -> u16 {
        // Allow at least one integer unit per dispatch when the speed limiter has settled.
        let minimum = (1.0 / DISPATCH_INTERVAL.as_secs_f64()).ceil();
        (f64::from(maximum) * self.slowdown(distance)).round().clamp(minimum, f64::from(maximum)) as u16
    }
}

impl Smoothing {
    pub fn reset(&mut self) {
        self.previous = None;
    }

    pub(crate) fn controlled(&mut self, offset: [f64; 2], velocity: [f64; 2], scale: [f64; 2], radial: f64,
        half_life_ms: f64, max_step: i32, at: u64, settings: &MotionConfig) -> [i32; 2] {
        let distance = offset[0].hypot(offset[1]);
        if distance <= radial { self.reset(); return [0, 0]; }
        let half_life = if settings.adaptive {
            let speed = velocity[0].hypot(velocity[1]);
            half_life_ms * (1.5 / (1.0 + speed * 4.0)).clamp(0.35, 1.5)
        } else { half_life_ms };
        let slowdown = settings.slowdown(distance);
        let command = std::array::from_fn(|axis| {
            if offset[axis] == 0.0 { return 0.0; }
            let derivative = match settings.algorithm { Algorithm::P => 0.0, Algorithm::Pd => settings.derivative_ms[axis] };
            let signed = offset[axis].signum();
            let lead = velocity[axis] * (derivative + settings.prediction_ms) * signed;
            (offset[axis].abs() + lead).clamp(0.0, offset[axis].abs() * 1.5) * signed * slowdown
        });
        let mut result = self.relative(command, scale, 0.0, half_life, max_step, at);
        if settings.slow_radius_px > 0.0 || settings.prediction_ms > 0.0 || matches!(settings.algorithm, Algorithm::Pd) {
            for axis in 0..2 {
                let error_budget = (offset[axis] * scale[axis]).abs().ceil().clamp(0.0, f64::from(max_step)) as i32;
                result[axis] = result[axis].clamp(-error_budget, error_budget);
            }
        }
        result
    }

    pub fn relative(&mut self, offset: [f64; 2], scale: [f64; 2], dead_zone: f64,
        half_life_ms: f64, max_step: i32, frame_at: u64) -> [i32; 2]
    {
        if offset[0].hypot(offset[1]) <= dead_zone {
            self.reset();
            return [0, 0];
        }
        let raw = [offset[0] * scale[0], offset[1] * scale[1]];
        let limit = max_step.clamp(1, KMBOX_AXIS_LIMIT);
        let value = match self.previous {
            Some((last_at, last)) if half_life_ms > 0.0 => {
                let elapsed = frame_at.saturating_sub(last_at).max(1) as f64;
                let decay = (-elapsed / half_life_ms).exp2();
                std::array::from_fn(|axis| {
                    if raw[axis] == 0.0 || raw[axis].is_sign_negative() != last[axis].is_sign_negative() {
                        raw[axis]
                    } else {
                        raw[axis] * (1.0 - decay) + last[axis] * decay
                    }
                })
            }
            _ => raw,
        };
        self.previous = Some((frame_at, value));
        value.map(|axis| (axis.round() as i32).clamp(-limit, limit))
    }
}

#[cfg(test)]
mod controller_tests {
    use super::*;

    #[test]
    fn low_speed_diagonal_preserves_both_directions_and_the_step_limit() {
        let mut dispatcher = Dispatcher::new(DryRunDriver);
        let now = Instant::now();
        let mut command = PendingMove { movement: RelativeMove { dx: -10, dy: 4 }, sequence: 1,
            frame_at: 0, created: now, expires_at_ms: 0, held: None, limits: Some((120, 4000)), target: None };
        for horizontal in [-10, 10] {
            command.movement.dx = horizontal;
            let moves: Vec<_> = (0..30).map(|_| dispatcher.limit_live(command)).collect();
            assert!(moves.iter().all(|movement| movement.dx * horizontal >= 0 && movement.dy >= 0 &&
                f64::from(movement.dx).hypot(f64::from(movement.dy)) <= 2.04));
            assert!(moves.iter().any(|movement| movement.dx != 0 && movement.dy > 0));
        }
        command.movement.dy = 0;
        assert_eq!(dispatcher.limit_live(command).dy, 0);
    }

    #[test]
    fn pd_brakes_on_approach_without_reversing_and_respects_dead_zone() {
        let p = MotionConfig { slow_radius_px: 40.0, ..MotionConfig::default() };
        let pd = MotionConfig { algorithm: Algorithm::Pd, derivative_ms: [16.0, 6.0],
            prediction_ms: 4.0, adaptive: true, ..p };
        let mut filter = Smoothing::default();
        let proportional = filter.controlled([24.0, -18.0], [-0.6, 0.3], [1.0; 2], 2.0, 0.0, 24, 16, &p);
        let derivative = filter.controlled([24.0, -18.0], [-0.6, 0.3], [1.0; 2], 2.0, 0.0, 24, 32, &pd);
        assert!(0 <= derivative[0] && derivative[0] < proportional[0]);
        assert!(proportional[1] < derivative[1] && derivative[1] <= 0);
        assert_eq!(filter.controlled([1.0, -1.0], [-3.0, 3.0], [1.0; 2], 2.0, 12.0, 24, 48, &pd), [0, 0]);
        assert!(pd.speed(16.0, 480) < pd.speed(80.0, 480));
    }
}
