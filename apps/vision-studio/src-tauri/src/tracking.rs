const MAX_PREDICTION_MS: u64 = 80;
const MAX_SPEED_PX_PER_MS: f64 = 4.0;

#[derive(Clone, Copy)]
pub(crate) struct Observation {
    pub class: u16,
    pub bounds: [f64; 4],
}

impl Observation {
    pub fn center(self) -> [f64; 2] {
        [(self.bounds[0] + self.bounds[2]) / 2.0, (self.bounds[1] + self.bounds[3]) / 2.0]
    }

    pub fn area(self) -> f64 {
        (self.bounds[2] - self.bounds[0]) * (self.bounds[3] - self.bounds[1])
    }

    fn association(self, last: Self, velocity: [f64; 2], age: u64) -> Option<f64> {
        if self.class != last.class { return None; }
        let center = self.center();
        let previous = last.center();
        let horizon = age.min(MAX_PREDICTION_MS) as f64;
        let shift = velocity.map(|axis| axis * horizon);
        let predicted = [last.bounds[0]+shift[0], last.bounds[1]+shift[1], last.bounds[2]+shift[0], last.bounds[3]+shift[1]];
        let current = self.bounds;
        let intersection = (current[2].min(predicted[2]) - current[0].max(predicted[0])).max(0.0)
            * (current[3].min(predicted[3]) - current[1].max(predicted[1])).max(0.0);
        let iou = intersection / (self.area() + last.area() - intersection);
        let distance = ((center[0] - previous[0] - shift[0]) / (last.bounds[2]-last.bounds[0]))
            .hypot((center[1] - previous[1] - shift[1]) / (last.bounds[3]-last.bounds[1]));
        let ratio = self.area() / last.area();
        (iou >= 0.20 || (distance <= 0.60 && (0.65..=1.55).contains(&ratio)))
            .then_some(iou - distance * 0.5)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Track {
    pub id: u64,
    observation: Observation,
    seen_at: u64,
    pub velocity: [f64; 2],
}

#[derive(Default)]
pub(crate) struct Tracker {
    tracks: Vec<Track>,
    next_id: u64,
    last_sequence: Option<u64>,
    last_matches: Vec<Track>,
}

impl Tracker {
    pub fn update(&mut self, observations: &[Observation], sequence: u64, at: u64, lost_wait_ms: u64) -> &[Track] {
        if self.last_sequence == Some(sequence) { return &self.last_matches; }
        self.last_sequence = Some(sequence);
        self.tracks.retain(|track| at.saturating_sub(track.seen_at) <= lost_wait_ms.max(150));
        // Bound work even if every frame contains a new set of unmatched detections.
        if self.tracks.len() > 600 {
            self.tracks.sort_unstable_by_key(|track| (std::cmp::Reverse(track.seen_at), track.id));
            self.tracks.truncate(600);
        }
        let mut matches = vec![None; observations.len()];
        let mut used = vec![false; self.tracks.len()];
        let mut edges = Vec::new();
        for (index, observation) in observations.iter().enumerate() {
            for (slot, track) in self.tracks.iter().enumerate() {
                let age = at.saturating_sub(track.seen_at);
                if let Some(rank) = observation.association(track.observation, track.velocity, age) {
                    edges.push((rank, slot, index));
                }
            }
        }
        edges.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)).then_with(|| a.2.cmp(&b.2)));
        for (_, slot, index) in edges {
            if used[slot] || matches[index].is_some() { continue; }
            let track = &mut self.tracks[slot];
            let observation = observations[index];
            let elapsed = at.saturating_sub(track.seen_at);
            if elapsed >= 4 {
                let before = track.observation.center();
                let after = observation.center();
                track.velocity = std::array::from_fn(|axis| {
                    let velocity = ((after[axis]-before[axis]) / elapsed as f64)
                        .clamp(-MAX_SPEED_PX_PER_MS, MAX_SPEED_PX_PER_MS);
                    let previous_velocity = track.velocity[axis];
                    if previous_velocity == 0.0 || (previous_velocity * velocity < 0.0 && velocity.abs() >= 0.15) {
                        velocity
                    } else {
                        previous_velocity * 0.65 + velocity * 0.35
                    }
                });
                track.observation = observation;
                track.seen_at = at;
            }
            used[slot] = true;
            matches[index] = Some(*track);
        }
        self.last_matches = matches.into_iter().zip(observations).map(|(track, &observation)| {
            track.unwrap_or_else(|| {
                self.next_id = sequence.max(self.next_id.saturating_add(1));
                let track = Track { id: self.next_id, observation, seen_at: at, velocity: [0.0; 2] };
                self.tracks.push(track);
                track
            })
        }).collect();
        &self.last_matches
    }
}
