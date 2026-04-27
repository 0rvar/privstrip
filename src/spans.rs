use crate::labels::{Boundary, LabelInfo};

#[derive(Debug, Clone)]
pub struct DetectedSpan {
    pub label: String,
    pub byte_start: usize,
    pub byte_end: usize,
    pub text: String,
}

/// Convert per-token label ids into (label, token_start, token_end_exclusive) tuples.
/// Forgiving: handles malformed BIES sequences by closing/opening as needed.
pub fn labels_to_token_spans(
    labels: &[usize],
    label_info: &LabelInfo,
) -> Vec<(String, usize, usize)> {
    let mut spans = Vec::new();
    let mut current_label: Option<String> = None;
    let mut start_idx: Option<usize> = None;
    let mut previous_idx: Option<usize> = None;

    for (token_idx, &label_id) in labels.iter().enumerate() {
        let span_label = label_info.span_label[label_id].clone();
        let boundary = label_info.boundary[label_id];

        if let Some(prev) = previous_idx
            && token_idx != prev + 1
            && let (Some(cl), Some(s)) = (current_label.as_ref(), start_idx)
        {
            spans.push((cl.clone(), s, prev + 1));
            current_label = None;
            start_idx = None;
        }

        if span_label.is_none() {
            previous_idx = Some(token_idx);
            continue;
        }

        let is_background = matches!(boundary, Boundary::Background);
        if is_background {
            if let (Some(cl), Some(s)) = (current_label.as_ref(), start_idx) {
                spans.push((cl.clone(), s, token_idx));
            }
            current_label = None;
            start_idx = None;
            previous_idx = Some(token_idx);
            continue;
        }

        let span_label = span_label.unwrap();
        match boundary {
            Boundary::S => {
                if let (Some(cl), Some(s), Some(prev)) = (
                    current_label.as_ref(),
                    start_idx,
                    previous_idx,
                ) {
                    spans.push((cl.clone(), s, prev + 1));
                }
                spans.push((span_label.clone(), token_idx, token_idx + 1));
                current_label = None;
                start_idx = None;
            }
            Boundary::B => {
                if let (Some(cl), Some(s), Some(prev)) = (
                    current_label.as_ref(),
                    start_idx,
                    previous_idx,
                ) {
                    spans.push((cl.clone(), s, prev + 1));
                }
                current_label = Some(span_label.clone());
                start_idx = Some(token_idx);
            }
            Boundary::I => {
                let need_open = current_label.as_ref() != Some(&span_label);
                if need_open {
                    if let (Some(cl), Some(s), Some(prev)) = (
                        current_label.as_ref(),
                        start_idx,
                        previous_idx,
                    ) {
                        spans.push((cl.clone(), s, prev + 1));
                    }
                    current_label = Some(span_label.clone());
                    start_idx = Some(token_idx);
                }
            }
            Boundary::E => {
                if current_label.as_ref() != Some(&span_label) || start_idx.is_none() {
                    if let (Some(cl), Some(s), Some(prev)) = (
                        current_label.as_ref(),
                        start_idx,
                        previous_idx,
                    ) {
                        spans.push((cl.clone(), s, prev + 1));
                    }
                    spans.push((span_label.clone(), token_idx, token_idx + 1));
                    current_label = None;
                    start_idx = None;
                } else {
                    let s = start_idx.unwrap();
                    spans.push((span_label.clone(), s, token_idx + 1));
                    current_label = None;
                    start_idx = None;
                }
            }
            Boundary::Background => unreachable!(),
        }
        previous_idx = Some(token_idx);
    }

    if let (Some(cl), Some(s), Some(prev)) = (current_label, start_idx, previous_idx) {
        spans.push((cl, s, prev + 1));
    }

    spans
}

/// Map token-index spans to byte-offset spans in the source text using per-token offsets.
/// `offsets[i] = (byte_start, byte_end)` for token i.
pub fn token_spans_to_byte_spans(
    spans: &[(String, usize, usize)],
    offsets: &[(usize, usize)],
) -> Vec<(String, usize, usize)> {
    let mut out = Vec::with_capacity(spans.len());
    for (label, ts, te) in spans {
        if !(*ts < *te && *te <= offsets.len()) {
            continue;
        }
        let bs = offsets[*ts].0;
        let be = offsets[*te - 1].1;
        if be <= bs {
            continue;
        }
        out.push((label.clone(), bs, be));
    }
    out
}

/// Trim leading and trailing ASCII whitespace inside each span. Drops empty results.
pub fn trim_whitespace(
    spans: &[(String, usize, usize)],
    text: &str,
) -> Vec<(String, usize, usize)> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(spans.len());
    for (label, start, end) in spans {
        let mut s = *start;
        let mut e = *end;
        while s < e && bytes[s].is_ascii_whitespace() {
            s += 1;
        }
        while e > s && bytes[e - 1].is_ascii_whitespace() {
            e -= 1;
        }
        if e > s {
            out.push((label.clone(), s, e));
        }
    }
    out
}

/// Within each label class, drop overlapping spans (keep longest, break ties by earliest start).
pub fn discard_overlapping_per_label(
    spans: Vec<(String, usize, usize)>,
) -> Vec<(String, usize, usize)> {
    let mut labels: Vec<String> = Vec::new();
    let mut buckets: Vec<Vec<(usize, usize)>> = Vec::new();
    for (label, s, e) in spans {
        let idx = match labels.iter().position(|l| l == &label) {
            Some(i) => i,
            None => {
                labels.push(label);
                buckets.push(Vec::new());
                labels.len() - 1
            }
        };
        buckets[idx].push((s, e));
    }
    let mut kept: Vec<(String, usize, usize)> = Vec::new();
    for (label, mut list) in labels.into_iter().zip(buckets.into_iter()) {
        // Sort by start asc, then by length desc.
        list.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
        let mut last_end: usize = 0;
        let mut started = false;
        for (s, e) in list {
            if !started || s >= last_end {
                kept.push((label.clone(), s, e));
                last_end = e;
                started = true;
            }
        }
    }
    kept.sort_by_key(|(_, s, _)| *s);
    kept
}

/// Greedily keep non-overlapping spans across all labels (left to right).
pub fn select_non_overlapping(
    spans: Vec<(String, usize, usize)>,
) -> Vec<(String, usize, usize)> {
    let mut sorted = spans;
    sorted.sort_by_key(|(_, s, _)| *s);
    let mut out: Vec<(String, usize, usize)> = Vec::new();
    let mut last_end: usize = 0;
    let mut started = false;
    for (label, s, e) in sorted {
        if !started || s >= last_end {
            out.push((label, s, e));
            last_end = e;
            started = true;
        }
    }
    out
}

/// Build the final list of detected spans from raw label ids.
pub fn extract_spans(
    labels: &[usize],
    offsets: &[(usize, usize)],
    text: &str,
    label_info: &LabelInfo,
) -> Vec<DetectedSpan> {
    let token_spans = labels_to_token_spans(labels, label_info);
    let byte_spans = token_spans_to_byte_spans(&token_spans, offsets);
    let trimmed = trim_whitespace(&byte_spans, text);
    let deduped = discard_overlapping_per_label(trimmed);
    let final_spans = select_non_overlapping(deduped);
    final_spans
        .into_iter()
        .filter_map(|(label, s, e)| {
            let snippet = text.get(s..e)?.to_string();
            Some(DetectedSpan {
                label,
                byte_start: s,
                byte_end: e,
                text: snippet,
            })
        })
        .collect()
}

/// Replace each span's bytes with `<LABEL>` placeholders, returning the redacted string.
pub fn redact(text: &str, spans: &[DetectedSpan]) -> String {
    let mut sorted: Vec<&DetectedSpan> = spans.iter().collect();
    sorted.sort_by_key(|s| s.byte_start);
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut cursor = 0usize;
    for s in sorted {
        if s.byte_start > cursor {
            out.push_str(std::str::from_utf8(&bytes[cursor..s.byte_start]).unwrap_or(""));
        }
        let placeholder = placeholder_for(&s.label);
        out.push_str(&placeholder);
        cursor = s.byte_end;
    }
    if cursor < bytes.len() {
        out.push_str(std::str::from_utf8(&bytes[cursor..]).unwrap_or(""));
    }
    out
}

fn placeholder_for(label: &str) -> String {
    let upper = label.to_uppercase();
    let mut buf = String::with_capacity(label.len() + 2);
    buf.push('<');
    let mut last_was_underscore = false;
    for ch in upper.chars() {
        if ch.is_ascii_alphanumeric() {
            buf.push(ch);
            last_was_underscore = false;
        } else if !last_was_underscore {
            buf.push('_');
            last_was_underscore = true;
        }
    }
    while buf.ends_with('_') {
        buf.pop();
    }
    buf.push('>');
    buf
}
