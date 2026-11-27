use super::{input, AutomationController, Duration, Instant};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

#[derive(Default)]
struct Totals {
    count: u64,
    decision_ms: f64,
    capture_to_candidate_ms: u64,
}

impl Totals {
    fn add(&mut self, capture_to_candidate_ms: u64, decision_ms: f64) {
        self.count += 1;
        self.capture_to_candidate_ms += capture_to_candidate_ms;
        self.decision_ms += decision_ms;
    }

    fn metrics(&self) -> Value {
        json!({"candidate_samples":self.count,
            "decision_avg_ms":(self.count > 0).then(|| self.decision_ms / self.count as f64),
            "capture_to_candidate_avg_ms":(self.count > 0).then(|| self.capture_to_candidate_ms as f64 / self.count as f64)})
    }
}

pub(super) fn verify() -> Result<(), String> {
    let mut gate = AutomationController::default();
    let profile = serde_json::from_value(json!({"mask":3,"enabled_classes":[0,1,2,3,4,5,6],
        "dead_zone_px":2.0,"smooth_half_life_ms":12.0})).map_err(|error| error.to_string())?;
    gate.configure(vec![profile])?;
    gate.request_stream(1);
    gate.request_model(1);
    let mut initially_armed = false;
    let mut failed = false;
    let mut last_sequence = None;
    let mut window = Totals::default();
    let mut total = Totals::default();
    let mut last_report = Instant::now();
    let stdout = io::stdout();
    let mut output = stdout.lock();
    for line in io::stdin().lock().lines() {
        let line = line.map_err(|error| error.to_string())?;
        let event: Value = serde_json::from_str(&line).map_err(|error| error.to_string())?;
        let detection = event["kind"] == "detections" && event["source"] == 1 && event["model"] == 1;
        if detection && !initially_armed && !failed {
            gate.buttons(&input(0, 0));
        }
        if detection && initially_armed && !failed {
            gate.buttons(&input(3, 0));
        }
        for update in [gate.worker(&event), gate.tick()].into_iter().flatten() {
            if update["state"] == "Fault" && !failed {
                failed = true;
                write_line(&mut output, json!({"event":"controller_fault","reason":update["reason"],"rearmed":false}))?;
            }
        }
        if detection && !initially_armed && !failed && gate.arm().is_ok() {
            gate.buttons(&input(3, 0));
            initially_armed = true;
        }
        if let Some((sequence, latency_ms, decision_ms)) = gate.target_sample() {
            if last_sequence != Some(sequence) {
                last_sequence = Some(sequence);
                window.add(latency_ms, decision_ms);
                total.add(latency_ms, decision_ms);
            }
        }
        if last_report.elapsed() >= Duration::from_secs(1) {
            write_line(&mut output, json!({"event":"controller_sample","timings":window.metrics(),
                "target":gate.target_update(), "dry_run_scheduled":gate.movement.submitted}))?;
            last_report = Instant::now();
            window = Totals::default();
        }
    }
    write_line(&mut output, json!({"event":"controller_summary","timings":total.metrics(),
        "dry_run_scheduled":gate.movement.submitted,"mode":"dry_run","input":"simulated buttons","fault":failed}))
}

fn write_line(output: &mut impl Write, value: Value) -> Result<(), String> {
    writeln!(output, "{value}").and_then(|_| output.flush()).map_err(|error| error.to_string())
}
