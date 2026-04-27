use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

use crate::labels::{Boundary, LabelInfo};

const NEG_INF: f32 = -1e9;

/// Viterbi transition biases. The on-disk format is at
/// `viterbi_calibration.json::operating_points.<name>.biases`. Default operating point
/// ships with all-zero biases, which is mathematically a no-op vs an unbiased
/// constraint-only decoder.
#[derive(Debug, Clone, Copy, Default, Deserialize)]
pub struct ViterbiBiases {
    #[serde(default)]
    pub transition_bias_background_stay: f32,
    #[serde(default)]
    pub transition_bias_background_to_start: f32,
    #[serde(default)]
    pub transition_bias_inside_to_continue: f32,
    #[serde(default)]
    pub transition_bias_inside_to_end: f32,
    #[serde(default)]
    pub transition_bias_end_to_background: f32,
    #[serde(default)]
    pub transition_bias_end_to_start: f32,
}

#[derive(Debug, Deserialize)]
struct CalibrationFile {
    operating_points: BTreeMap<String, OperatingPoint>,
}

#[derive(Debug, Deserialize)]
struct OperatingPoint {
    biases: ViterbiBiases,
}

impl ViterbiBiases {
    /// Load the named operating point from a calibration JSON file.
    pub fn from_calibration_file(path: &Path, operating_point: &str) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("read {}", path.display()))?;
        let parsed: CalibrationFile = serde_json::from_str(&raw)
            .with_context(|| format!("parse {}", path.display()))?;
        parsed
            .operating_points
            .get(operating_point)
            .map(|op| op.biases)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "operating point {operating_point:?} not found in {}",
                    path.display()
                )
            })
    }

    /// Look up the bias for one (prev, next) edge per OPF's `_transition_bias` table.
    /// `prev_is_bg` and `next_is_bg` are pre-computed background checks; tags refer to
    /// the BIES boundary of each side.
    fn for_edge(
        &self,
        prev_boundary: Boundary,
        prev_is_bg: bool,
        next_boundary: Boundary,
        next_is_bg: bool,
        same_span: bool,
    ) -> f32 {
        if prev_is_bg {
            if next_is_bg {
                return self.transition_bias_background_stay;
            }
            if matches!(next_boundary, Boundary::B | Boundary::S) {
                return self.transition_bias_background_to_start;
            }
            return 0.0;
        }
        match prev_boundary {
            Boundary::B | Boundary::I => {
                if same_span && matches!(next_boundary, Boundary::I) {
                    self.transition_bias_inside_to_continue
                } else if same_span && matches!(next_boundary, Boundary::E) {
                    self.transition_bias_inside_to_end
                } else {
                    0.0
                }
            }
            Boundary::E | Boundary::S => {
                if next_is_bg {
                    self.transition_bias_end_to_background
                } else if matches!(next_boundary, Boundary::B | Boundary::S) {
                    self.transition_bias_end_to_start
                } else {
                    0.0
                }
            }
            Boundary::Background => 0.0, // unreachable: covered by prev_is_bg above
        }
    }
}

pub struct ViterbiDecoder {
    num_classes: usize,
    start_scores: Vec<f32>,
    end_scores: Vec<f32>,
    transition_scores: Vec<f32>, // [num_classes * num_classes], row-major (prev, next)
}

impl ViterbiDecoder {
    pub fn with_biases(label_info: &LabelInfo, biases: ViterbiBiases) -> Self {
        let n = label_info.num_classes();
        let bg = label_info.background_idx;

        let mut start_scores = vec![NEG_INF; n];
        let mut end_scores = vec![NEG_INF; n];
        for i in 0..n {
            // Start: background or B/S
            if i == bg
                || matches!(label_info.boundary[i], Boundary::B | Boundary::S)
            {
                start_scores[i] = 0.0;
            }
            // End: background or E/S
            if i == bg
                || matches!(label_info.boundary[i], Boundary::E | Boundary::S)
            {
                end_scores[i] = 0.0;
            }
        }

        let mut transition_scores = vec![NEG_INF; n * n];
        for prev in 0..n {
            for next in 0..n {
                if !is_valid_transition(label_info, prev, next) {
                    continue;
                }
                let prev_is_bg = prev == bg;
                let next_is_bg = next == bg;
                let same_span = label_info.span_label[prev] == label_info.span_label[next];
                transition_scores[prev * n + next] = biases.for_edge(
                    label_info.boundary[prev],
                    prev_is_bg,
                    label_info.boundary[next],
                    next_is_bg,
                    same_span,
                );
            }
        }

        Self {
            num_classes: n,
            start_scores,
            end_scores,
            transition_scores,
        }
    }

    /// Decode a `[seq_len, num_classes]` log-probability matrix into a label-id sequence.
    pub fn decode(&self, log_probs: &[f32], seq_len: usize) -> Vec<usize> {
        let n = self.num_classes;
        if seq_len == 0 {
            return Vec::new();
        }
        debug_assert_eq!(log_probs.len(), seq_len * n);

        let mut scores = vec![0f32; n];
        for i in 0..n {
            scores[i] = log_probs[i] + self.start_scores[i];
        }

        let mut backpointers = vec![0u32; (seq_len.saturating_sub(1)) * n];
        let mut next_scores = vec![0f32; n];

        for step in 1..seq_len {
            let token_lp = &log_probs[step * n..(step + 1) * n];
            for next in 0..n {
                let mut best_score = NEG_INF;
                let mut best_prev = 0usize;
                for prev in 0..n {
                    let candidate = scores[prev] + self.transition_scores[prev * n + next];
                    if candidate > best_score {
                        best_score = candidate;
                        best_prev = prev;
                    }
                }
                next_scores[next] = best_score + token_lp[next];
                backpointers[(step - 1) * n + next] = best_prev as u32;
            }
            std::mem::swap(&mut scores, &mut next_scores);
        }

        // Apply end-state biases.
        let mut final_scores = scores.clone();
        let mut any_finite = false;
        for i in 0..n {
            final_scores[i] += self.end_scores[i];
            if final_scores[i].is_finite() && final_scores[i] > NEG_INF / 2.0 {
                any_finite = true;
            }
        }

        if !any_finite {
            // Fallback: argmax per token (matches reference behavior on degenerate inputs).
            let mut path = Vec::with_capacity(seq_len);
            for step in 0..seq_len {
                let token_lp = &log_probs[step * n..(step + 1) * n];
                let mut best = 0usize;
                let mut best_v = f32::NEG_INFINITY;
                for (i, &v) in token_lp.iter().enumerate() {
                    if v > best_v {
                        best_v = v;
                        best = i;
                    }
                }
                path.push(best);
            }
            return path;
        }

        let mut last = 0usize;
        let mut last_v = f32::NEG_INFINITY;
        for (i, &v) in final_scores.iter().enumerate() {
            if v > last_v {
                last_v = v;
                last = i;
            }
        }

        let mut path = vec![0usize; seq_len];
        path[seq_len - 1] = last;
        for step in (0..seq_len - 1).rev() {
            last = backpointers[step * n + last] as usize;
            path[step] = last;
        }
        path
    }
}

fn is_valid_transition(label_info: &LabelInfo, prev: usize, next: usize) -> bool {
    let bg = label_info.background_idx;
    let next_is_bg = next == bg;
    let prev_is_bg = prev == bg;

    if prev_is_bg {
        return next_is_bg
            || matches!(label_info.boundary[next], Boundary::B | Boundary::S);
    }

    match label_info.boundary[prev] {
        Boundary::Background => next_is_bg
            || matches!(label_info.boundary[next], Boundary::B | Boundary::S),
        Boundary::E | Boundary::S => {
            next_is_bg || matches!(label_info.boundary[next], Boundary::B | Boundary::S)
        }
        Boundary::B | Boundary::I => {
            // Must continue same span with I or E of the same entity type.
            if !matches!(label_info.boundary[next], Boundary::I | Boundary::E) {
                return false;
            }
            label_info.span_label[prev] == label_info.span_label[next]
        }
    }
}
