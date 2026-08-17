//! The detection pipeline: tokenize → forward → decode → extract spans.
//!
//! `Engine` is `&self`-only and every mutable bit of state behind it is
//! internally synchronized, so one engine can be shared across threads. Note
//! that the forward pass parallelizes internally with rayon, so a caller that
//! wants throughput should give each engine its own thread rather than calling
//! one engine from many.

use std::collections::HashMap;

use anyhow::{Context, Result};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::ops;
use clap::ValueEnum;
use tokenizers::Tokenizer;

use crate::config::ModelConfig;
use crate::labels::LabelInfo;
use crate::model::Transformer;
use crate::spans::{DetectedSpan, extract_spans};
use crate::timing::{Stage, time_stage};
use crate::viterbi::{ViterbiBiases, ViterbiDecoder};

/// Operating point every checkpoint in this family is expected to define, and
/// the one used when a request or CLI invocation doesn't name one.
pub const DEFAULT_OPERATING_POINT: &str = "default";

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum DecoderMode {
    /// Constraint-aware decoding: rejects malformed BIES sequences.
    Viterbi,
    /// Independent per-token argmax. Matches transformers.js's stock pipeline.
    Argmax,
}

impl DecoderMode {
    /// Parse the name used by the `decoder` field of an HTTP request. The
    /// accepted spellings match the CLI's `--decoder` values.
    pub fn from_wire_name(s: &str) -> Option<Self> {
        match s {
            "viterbi" => Some(Self::Viterbi),
            "argmax" => Some(Self::Argmax),
            _ => None,
        }
    }
}

pub struct Engine {
    tokenizer: Tokenizer,
    model: Transformer,
    /// One decoder per operating point declared by the checkpoint's
    /// `viterbi_calibration.json`. Built up-front because a request picks its
    /// operating point per call and constructing a decoder is not free.
    decoders: HashMap<String, ViterbiDecoder>,
    label_info: LabelInfo,
    device: Device,
}

impl Engine {
    pub fn load(model_dir: &std::path::Path, device: Device) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let tokenizer_path = model_dir.join("tokenizer.json");
        let weights_path = model_dir.join("model.safetensors");
        let calibration_path = model_dir.join("viterbi_calibration.json");

        let config = ModelConfig::from_file(&config_path)?;
        let label_info = LabelInfo::from_config(&config_path)?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
        let model = Transformer::load(&weights_path, config, device.clone())?;
        let decoders = load_decoders(&calibration_path, &label_info)?;

        Ok(Self {
            tokenizer,
            model,
            decoders,
            label_info,
            device,
        })
    }

    /// The decoder for a named operating point, or an error naming the ones that
    /// exist. Callers use this to validate an operating point up-front, before
    /// spending a forward pass to discover the name was wrong.
    pub fn decoder_for(&self, operating_point: &str) -> Result<&ViterbiDecoder> {
        self.decoders.get(operating_point).ok_or_else(|| {
            let mut known: Vec<&str> = self.decoders.keys().map(String::as_str).collect();
            known.sort_unstable();
            anyhow::anyhow!(
                "unknown operating point {operating_point:?} (known: {})",
                known.join(", ")
            )
        })
    }

    /// Operating points this checkpoint can decode with, sorted for stable output.
    pub fn operating_points(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.decoders.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn model(&self) -> &Transformer {
        &self.model
    }

    pub fn label_info(&self) -> &LabelInfo {
        &self.label_info
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// How many tokens `text` encodes to, without running the model.
    ///
    /// Exists so a server can reject an oversized item before the forward pass,
    /// whose attention tensors grow with the square of the token count. Kept
    /// separate from `detect` rather than folded into it so the one-shot CLI
    /// paths stay exactly as they were.
    pub fn count_tokens(&self, text: &str) -> Result<usize> {
        if text.is_empty() {
            return Ok(0);
        }
        let encoding = time_stage(Stage::Tokenize, || {
            self.tokenizer
                .encode(text, false)
                .map_err(|e| anyhow::anyhow!("encode: {e}"))
        })?;
        Ok(encoding.get_ids().len())
    }

    /// Tokenize, run the model, decode BIES, and extract final spans.
    pub fn detect(
        &self,
        text: &str,
        decoder_mode: DecoderMode,
        operating_point: &str,
    ) -> Result<DetectionResult> {
        if text.is_empty() {
            return Ok(DetectionResult::default());
        }
        let decoder = match decoder_mode {
            DecoderMode::Viterbi => Some(self.decoder_for(operating_point)?),
            DecoderMode::Argmax => None,
        };
        let encoding = time_stage(Stage::Tokenize, || {
            self.tokenizer
                .encode(text, false)
                .map_err(|e| anyhow::anyhow!("encode: {e}"))
        })?;
        let token_ids = encoding.get_ids().to_vec();
        let offsets: Vec<(usize, usize)> = encoding.get_offsets().to_vec();
        let tokens_count = token_ids.len();
        if tokens_count == 0 {
            return Ok(DetectionResult::default());
        }

        let tokens = Tensor::from_vec(token_ids.clone(), tokens_count, &self.device)?;
        let logits = time_stage(Stage::Forward, || self.model.forward(&tokens))?;
        let log_probs_v = time_stage(Stage::Logits, || -> Result<Vec<f32>> {
            let log_probs = ops::log_softmax(&logits.to_dtype(DType::F32)?, D::Minus1)?;
            let flat = log_probs.flatten_all()?;
            // The to_vec1 call forces a device->host sync on Metal. Time it
            // separately so we can see the sync cost vs the softmax math.
            time_stage(Stage::LogitsSync, || flat.to_vec1::<f32>().map_err(Into::into))
        })?;

        let label_path = time_stage(Stage::Decode, || match decoder {
            Some(d) => d.decode(&log_probs_v, tokens_count),
            None => argmax_decode(&log_probs_v, tokens_count, self.label_info.num_classes()),
        });
        let spans = time_stage(Stage::SpanExtract, || {
            extract_spans(&label_path, &offsets, text, &self.label_info)
        });
        Ok(DetectionResult { spans, tokens: tokens_count })
    }
}

#[derive(Default)]
pub struct DetectionResult {
    pub spans: Vec<DetectedSpan>,
    pub tokens: usize,
}

/// Build one decoder per operating point the checkpoint declares.
///
/// Mirrors opf's `discover_default_viterbi_calibration_path`: the calibration
/// file next to the weights is authoritative when present, and a checkpoint
/// without one decodes with all-zero biases (constraint-only) under the
/// conventional `default` name.
///
/// Building every operating point eagerly is deliberate, not thoroughness for
/// its own sake: a malformed calibration entry must fail at load time, while
/// there is still an operator watching a startup that hasn't bound a port yet,
/// rather than surfacing as a per-request error once the model is serving
/// traffic. The consequence to be aware of is that the set of valid operating
/// points is fixed by the file — so asking for anything other than `default`
/// against a model directory with no calibration file is an error by design,
/// not an oversight.
fn load_decoders(
    calibration_path: &std::path::Path,
    label_info: &LabelInfo,
) -> Result<HashMap<String, ViterbiDecoder>> {
    let mut decoders = HashMap::new();
    if !calibration_path.exists() {
        decoders.insert(
            DEFAULT_OPERATING_POINT.to_string(),
            ViterbiDecoder::with_biases(label_info, ViterbiBiases::default()),
        );
        return Ok(decoders);
    }

    let raw = std::fs::read_to_string(calibration_path)
        .with_context(|| format!("read {}", calibration_path.display()))?;
    let parsed: serde_json::Value = serde_json::from_str(&raw)
        .with_context(|| format!("parse {}", calibration_path.display()))?;
    let names: Vec<String> = parsed
        .get("operating_points")
        .and_then(|v| v.as_object())
        .map(|m| m.keys().cloned().collect())
        .unwrap_or_default();
    anyhow::ensure!(
        !names.is_empty(),
        "no operating points declared in {}",
        calibration_path.display()
    );
    for name in names {
        let biases = ViterbiBiases::from_calibration_file(calibration_path, &name)?;
        decoders.insert(name, ViterbiDecoder::with_biases(label_info, biases));
    }
    Ok(decoders)
}

/// Per-item result envelope shared by the CLI's `stream` mode and the HTTP
/// service's detect endpoint, so the two protocols never drift apart.
pub fn detection_json(
    id: serde_json::Value,
    result: &DetectionResult,
    elapsed_us: u64,
) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "spans": result.spans.iter().map(|s| serde_json::json!({
            "label": s.label,
            "byte_start": s.byte_start,
            "byte_end": s.byte_end,
            "text": s.text,
        })).collect::<Vec<_>>(),
        "tokens": result.tokens,
        "elapsed_us": elapsed_us,
    })
}

pub fn detection_error_json(id: serde_json::Value, message: String) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "error": message,
    })
}

/// Independent per-token argmax over a `[seq_len, n]` log-probability matrix.
pub fn argmax_decode(log_probs: &[f32], seq_len: usize, n: usize) -> Vec<usize> {
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
    path
}

/// CPU unless `use_metal`. Metal is opt-in because some sandboxes cannot
/// initialize it, and because candle ships a stub Metal backend when the
/// `apple` feature is off — in which case this errors rather than silently
/// falling back.
pub fn pick_device(use_metal: bool) -> Result<Device> {
    if use_metal {
        return Ok(Device::new_metal(0)?);
    }
    Ok(Device::Cpu)
}
