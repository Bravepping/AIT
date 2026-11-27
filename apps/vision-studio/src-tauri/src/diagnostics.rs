use crate::control::now_ms;
use serde_json::{json, Value};
use std::time::Instant;

#[derive(Default)]
pub struct StreamDiagnostics {
    source: u32,
    nominal: f64,
    frame_time: Option<u64>,
    raw_pending: Option<u64>,
    prepared_pending: Option<u64>,
    input_dip: Option<Instant>,
    inference_dip: Option<Instant>,
}

impl StreamDiagnostics {
    pub fn observe(&mut self, event: &Value, source: u32, target: Option<(u64, u64)>) -> Option<Value> {
        if event["source"] != source {
            return None;
        }
        if self.source != source || event["state"] == "lost" {
            *self = Self { source, ..Self::default() };
        }
        let (metric, fps, threshold, previous) = match event["kind"].as_str()? {
            "ndi_health" => {
                self.frame_time = event["frame_at"].as_u64();
                self.raw_pending = event["raw_pending"].as_u64();
                self.prepared_pending = event["prepared_pending"].as_u64();
                return None;
            }
            "stream" => {
                self.nominal = event["nominal"].as_f64()?;
                ("input", event["fps"].as_f64()?, self.nominal * 0.9, &mut self.input_dip)
            }
            "inference" if event["warm_samples"].as_u64().unwrap_or(0) >= 120 => {
                ("inference", event["inference_fps"].as_f64()?, self.nominal * 0.75, &mut self.inference_dip)
            }
            _ => return None,
        };
        if !threshold.is_finite() || threshold < 1.0 || !fps.is_finite() {
            return None;
        }
        let duration = if fps < threshold {
            if previous.is_some() {
                return None;
            }
            *previous = Some(Instant::now());
            None
        } else {
            Some(previous.take()?.elapsed().as_millis())
        };
        let now = now_ms();
        Some(json!({"kind":"rate_dip","source":source,"metric":metric,"fps":fps,"threshold":threshold,
            "duration_ms":duration,"frame_age_ms":self.frame_time.and_then(|at| now.checked_sub(at)),
            "raw_pending":self.raw_pending, "prepared_pending":self.prepared_pending,
            "target_sequence":target.map(|(_, frame)| frame),
            "target_updated_at_ms":target.map(|(at, _)| at),
            "target_age_ms":target.and_then(|(at, _)| now.checked_sub(at))}))
    }
}
