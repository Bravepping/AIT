const MAX_PREDICTION_MS: u64 = 80;
const MAX_SPEED_PX_PER_MS: f64 = 4.0;
const CONTINUOUS_GAP_MS: u64 = 150;
const REACQUIRE_WINDOW_MS: u64 = 320;
const REACQUIRE_MARGIN: f64 = 0.16;

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

    fn association(self, last: Self, velocity: [f64; 2], age: u64, reacquire: bool) -> Option<f64> {
        if self.class != last.class { return None; }
        let center = self.center();
        let previous = last.center();
        let horizon = (if reacquire { age } else { age.min(MAX_PREDICTION_MS) }) as f64;
        let shift = velocity.map(|axis| axis * horizon);
        let predicted = [last.bounds[0]+shift[0], last.bounds[1]+shift[1], last.bounds[2]+shift[0], last.bounds[3]+shift[1]];
        let current = self.bounds;
        let intersection = (current[2].min(predicted[2]) - current[0].max(predicted[0])).max(0.0)
            * (current[3].min(predicted[3]) - current[1].max(predicted[1])).max(0.0);
        let iou = intersection / (self.area() + last.area() - intersection);
        let distance = ((center[0] - previous[0] - shift[0]) / (last.bounds[2]-last.bounds[0]))
            .hypot((center[1] - previous[1] - shift[1]) / (last.bounds[3]-last.bounds[1]));
        let ratio = self.area() / last.area();
        if reacquire {
            let speed = velocity[0].hypot(velocity[1]);
            let travel = [center[0] - previous[0], center[1] - previous[1]];
            let displacement = travel[0].hypot(travel[1]);
            let direction = if speed >= 0.05 && displacement >= 2.0 {
                (travel[0] * velocity[0] + travel[1] * velocity[1]) / (speed * displacement)
            } else { 0.0 };
            if !(0.65..=1.55).contains(&ratio) { return None; }
            let plausible = if speed >= 0.05 {
                direction >= 0.5 && distance <= 0.45 && iou >= 0.30
            } else {
                distance <= 0.24 && iou >= 0.55
            };
            return plausible.then_some(iou - distance * 0.5 + direction * 0.10);
        }
        (iou >= 0.20 || (distance <= 0.60 && (0.65..=1.55).contains(&ratio)))
            .then_some(iou - distance * 0.5)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Track {
    pub id: u64,
    observation: Observation,
    seen_at: u64,
    seen_sequence: u64,
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
    pub fn update(&mut self, observations: &[Observation], sequence: u64, at: u64) -> &[Track] {
        if self.last_sequence == Some(sequence) { return &self.last_matches; }
        let previous_sequence = self.last_sequence;
        self.last_sequence = Some(sequence);
        self.tracks.retain(|track| at.saturating_sub(track.seen_at) <= REACQUIRE_WINDOW_MS);
        // Bound work even if every frame contains a new set of unmatched detections.
        if self.tracks.len() > 600 {
            self.tracks.sort_unstable_by_key(|track| (std::cmp::Reverse(track.seen_at), track.id));
            self.tracks.truncate(600);
        }
        let mut matches = vec![None; observations.len()];
        let mut used = vec![false; self.tracks.len()];
        let mut edges = Vec::new();
        let mut dormant = Vec::new();
        for (index, observation) in observations.iter().enumerate() {
            for (slot, track) in self.tracks.iter().enumerate() {
                let age = at.saturating_sub(track.seen_at);
                let reacquire = previous_sequence != Some(track.seen_sequence) || age > CONTINUOUS_GAP_MS;
                if let Some(rank) = observation.association(track.observation, track.velocity, age, reacquire) {
                    let candidates = if reacquire { &mut dormant } else { &mut edges };
                    candidates.push((rank, slot, index));
                }
            }
        }
        edges.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)).then_with(|| a.2.cmp(&b.2)));
        for (_, slot, index) in edges {
            if used[slot] || matches[index].is_some() { continue; }
            used[slot] = true;
            matches[index] = Some(slot);
        }
        // A visible track claims its detection first. A dormant track must be
        // the clear best match on both sides before it can reclaim an ID.
        if !dormant.is_empty() {
            dormant.retain(|&(_, slot, index)| !used[slot] && matches[index].is_none());
            let mut by_track = vec![[f64::NEG_INFINITY; 2]; self.tracks.len()];
            let mut by_detection = vec![[f64::NEG_INFINITY; 2]; observations.len()];
            for &(rank, slot, index) in &dormant {
                record_rank(&mut by_track[slot], rank);
                record_rank(&mut by_detection[index], rank);
            }
            for (rank, slot, index) in dormant {
                let decisive = |[best, runner_up]: [f64; 2]| rank == best && best - runner_up >= REACQUIRE_MARGIN;
                if decisive(by_track[slot]) && decisive(by_detection[index]) {
                    matches[index] = Some(slot);
                }
            }
        }
        self.last_matches = matches.into_iter().zip(observations).map(|(track, &observation)| {
            if let Some(slot) = track {
                let track = &mut self.tracks[slot];
                track.observe(observation, sequence, at);
                *track
            } else {
                self.next_id = sequence.max(self.next_id.saturating_add(1));
                let track = Track { id: self.next_id, observation, seen_at: at, seen_sequence: sequence, velocity: [0.0; 2] };
                self.tracks.push(track);
                track
            }
        }).collect();
        &self.last_matches
    }
}

impl Track {
    fn observe(&mut self, observation: Observation, sequence: u64, at: u64) {
        let elapsed = at.saturating_sub(self.seen_at);
        self.seen_sequence = sequence;
        if elapsed < 4 { return; }
        let before = self.observation.center();
        let after = observation.center();
        self.velocity = std::array::from_fn(|axis| {
            let velocity = ((after[axis] - before[axis]) / elapsed as f64)
                .clamp(-MAX_SPEED_PX_PER_MS, MAX_SPEED_PX_PER_MS);
            let previous = self.velocity[axis];
            if previous == 0.0 || (previous * velocity < 0.0 && velocity.abs() >= 0.15) {
                velocity
            } else {
                previous * 0.65 + velocity * 0.35
            }
        });
        self.observation = observation;
        self.seen_at = at;
    }
}

fn record_rank([best, runner_up]: &mut [f64; 2], rank: f64) {
    if rank >= *best {
        *runner_up = *best;
        *best = rank;
    } else if rank > *runner_up {
        *runner_up = rank;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(x: f64, class: u16) -> Observation {
        Observation { class, bounds: [x - 22.0, 100.0, x + 22.0, 160.0] }
    }

    #[test]
    fn missed_track_uses_motion_without_reusing_an_expired_id() {
        let mut tracker = Tracker::default();
        let id = tracker.update(&[at(200.0, 0)], 1, 0)[0].id;
        tracker.update(&[at(212.0, 0)], 2, 40);
        assert!(tracker.update(&[], 3, 123).is_empty());
        assert!(tracker.update(&[], 4, 206).is_empty());
        let matches = tracker.update(&[at(198.0, 0), at(272.0, 1), at(272.0, 0)], 5, 240);
        assert_ne!(matches[0].id, id);
        assert_ne!(matches[1].id, id);
        assert_eq!(matches[2].id, id);
        assert!(tracker.update(&[], 6, 400).is_empty());
        assert_ne!(tracker.update(&[at(368.3, 0)], 7, 561)[0].id, id);
    }

    #[test]
    fn dormant_match_requires_clear_identity_and_never_steals_visible_track() {
        let mut tracker = Tracker::default();
        let first = tracker.update(&[at(200.0, 0), at(260.0, 0)], 1, 0);
        let ids = [first[0].id, first[1].id];
        tracker.update(&[at(210.0, 0), at(250.0, 0)], 2, 40);
        tracker.update(&[], 3, 80);
        let ambiguous = tracker.update(&[at(230.0, 0)], 4, 120)[0].id;
        assert!(!ids.contains(&ambiguous));
        assert_eq!(tracker.update(&[at(235.0, 0)], 5, 140)[0].id, ambiguous);

        let mut tracker = Tracker::default();
        let id = tracker.update(&[at(300.0, 0)], 1, 0)[0].id;
        tracker.update(&[], 2, 15);
        let detections = tracker.update(&[at(298.0, 0), at(302.0, 0)], 3, 30);
        assert!(detections.iter().all(|track| track.id != id));
    }
}
