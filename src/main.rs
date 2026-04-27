mod config;
mod labels;
mod model;
mod spans;
mod viterbi;

use std::io::{BufRead, Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::ops;
use clap::{Parser, ValueEnum};
use tokenizers::Tokenizer;

use crate::config::ModelConfig;
use crate::labels::LabelInfo;
use crate::model::Transformer;
use crate::spans::{DetectedSpan, extract_spans, redact};
use crate::viterbi::ViterbiDecoder;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DecoderMode {
    /// Constraint-aware decoding: rejects malformed BIES sequences.
    Viterbi,
    /// Independent per-token argmax. Matches transformers.js's stock pipeline.
    Argmax,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Mode {
    /// Print PII locations and exit 1 if any are found.
    Check,
    /// Print the input with PII replaced by <LABEL> placeholders.
    Redact,
    /// Print a JSON list of detected PII spans.
    List,
    /// Print per-token predictions for debugging.
    Debug,
    /// Read JSON-lines (`{"id":..., "text":...}`) from stdin, emit matching `{"id":..., "spans":[...]}`.
    Stream,
}

#[derive(Debug, Parser)]
#[command(
    name = "privstrip",
    about = "Detect personally identifiable information in text using openai/privacy-filter."
)]
struct Cli {
    /// What to do with detected PII.
    #[arg(value_enum, default_value_t = Mode::Check)]
    mode: Mode,
    /// Read from this file. If omitted (and --text is not set), read stdin.
    #[arg(short = 'f', long)]
    file: Option<PathBuf>,
    /// Use this string literal as input.
    #[arg(short = 't', long, conflicts_with = "file")]
    text: Option<String>,
    /// Directory containing model.safetensors, config.json, tokenizer.json.
    #[arg(short = 'm', long, default_value = "models")]
    model_dir: PathBuf,
    /// Use Metal (Apple GPU) instead of CPU. Off by default because some sandboxes can't init Metal.
    #[arg(long)]
    metal: bool,
    /// Decoding strategy applied to per-token logits.
    #[arg(long, value_enum, default_value_t = DecoderMode::Viterbi)]
    decoder: DecoderMode,
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<ExitCode> {
    let cli = Cli::parse();
    let device = pick_device(cli.metal)?;
    let engine = Engine::load(&cli.model_dir, device, cli.decoder)?;

    match cli.mode {
        Mode::Stream => run_stream(&engine),
        Mode::Debug => {
            let text = read_input(&cli)?;
            run_debug(&engine, &text)
        }
        mode => {
            let text = read_input(&cli)?;
            let result = engine.detect(&text)?;
            emit_and_exit(mode, &text, &result.spans)
        }
    }
}

struct Engine {
    tokenizer: Tokenizer,
    model: Transformer,
    decoder: ViterbiDecoder,
    decoder_mode: DecoderMode,
    label_info: LabelInfo,
    device: Device,
}

impl Engine {
    fn load(model_dir: &std::path::Path, device: Device, decoder_mode: DecoderMode) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let tokenizer_path = model_dir.join("tokenizer.json");
        let weights_path = model_dir.join("model.safetensors");

        let config = ModelConfig::from_file(&config_path)?;
        let label_info = LabelInfo::from_config(&config_path)?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
        let model = Transformer::load(&weights_path, config, device.clone())?;
        let decoder = ViterbiDecoder::new(&label_info);
        Ok(Self {
            tokenizer,
            model,
            decoder,
            decoder_mode,
            label_info,
            device,
        })
    }

    /// Tokenize, run the model, decode BIES with Viterbi, and extract final spans.
    fn detect(&self, text: &str) -> Result<DetectionResult> {
        if text.is_empty() {
            return Ok(DetectionResult::default());
        }
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        let token_ids = encoding.get_ids().to_vec();
        let offsets: Vec<(usize, usize)> = encoding.get_offsets().to_vec();
        let tokens_count = token_ids.len();
        if tokens_count == 0 {
            return Ok(DetectionResult::default());
        }

        let tokens = Tensor::from_vec(token_ids.clone(), tokens_count, &self.device)?;
        let logits = self.model.forward(&tokens)?;
        let log_probs = ops::log_softmax(&logits.to_dtype(DType::F32)?, D::Minus1)?;
        let log_probs_v: Vec<f32> = log_probs.flatten_all()?.to_vec1()?;

        let label_path = match self.decoder_mode {
            DecoderMode::Viterbi => self.decoder.decode(&log_probs_v, tokens_count),
            DecoderMode::Argmax => argmax_decode(&log_probs_v, tokens_count, self.label_info.num_classes()),
        };
        let spans = extract_spans(&label_path, &offsets, text, &self.label_info);
        Ok(DetectionResult { spans, tokens: tokens_count })
    }
}

#[derive(Default)]
struct DetectionResult {
    spans: Vec<DetectedSpan>,
    tokens: usize,
}

fn run_stream(engine: &Engine) -> Result<ExitCode> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut reader = stdin.lock();
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            break;
        }
        let trimmed = line.trim_end_matches(['\n', '\r']);
        if trimmed.is_empty() {
            continue;
        }
        let response = process_stream_line(engine, trimmed);
        serde_json::to_writer(&mut out, &response)?;
        out.write_all(b"\n")?;
        out.flush()?;
    }
    Ok(ExitCode::SUCCESS)
}

fn process_stream_line(engine: &Engine, line: &str) -> serde_json::Value {
    #[derive(serde::Deserialize)]
    struct In {
        id: serde_json::Value,
        text: String,
    }
    let parsed: Result<In, _> = serde_json::from_str(line);
    let (id, text) = match parsed {
        Ok(v) => (v.id, v.text),
        Err(e) => {
            return serde_json::json!({
                "id": null,
                "error": format!("invalid input json: {e}"),
            });
        }
    };
    let started = std::time::Instant::now();
    match engine.detect(&text) {
        Ok(res) => {
            let elapsed_us = started.elapsed().as_micros() as u64;
            serde_json::json!({
                "id": id,
                "spans": res.spans.iter().map(|s| serde_json::json!({
                    "label": s.label,
                    "byte_start": s.byte_start,
                    "byte_end": s.byte_end,
                    "text": s.text,
                })).collect::<Vec<_>>(),
                "tokens": res.tokens,
                "elapsed_us": elapsed_us,
            })
        }
        Err(e) => serde_json::json!({
            "id": id,
            "error": format!("{e:#}"),
        }),
    }
}

fn run_debug(engine: &Engine, text: &str) -> Result<ExitCode> {
    if text.is_empty() {
        return Ok(ExitCode::SUCCESS);
    }
    let encoding = engine
        .tokenizer
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
    let token_ids = encoding.get_ids().to_vec();
    let offsets: Vec<(usize, usize)> = encoding.get_offsets().to_vec();
    if token_ids.is_empty() {
        return Ok(ExitCode::SUCCESS);
    }

    let tokens = Tensor::from_vec(token_ids.clone(), token_ids.len(), &engine.device)?;
    let logits = engine.model.forward(&tokens)?;
    let log_probs_t = ops::log_softmax(&logits.to_dtype(DType::F32)?, D::Minus1)?;
    let log_probs_v: Vec<f32> = log_probs_t.flatten_all()?.to_vec1()?;
    let n = engine.label_info.num_classes();
    let label_path = match engine.decoder_mode {
        DecoderMode::Viterbi => engine.decoder.decode(&log_probs_v, token_ids.len()),
        DecoderMode::Argmax => argmax_decode(&log_probs_v, token_ids.len(), n),
    };
    for (i, &tid) in token_ids.iter().enumerate() {
        let (bs, be) = offsets[i];
        let token_text = text.get(bs..be).unwrap_or("");
        let label_id = label_path[i];
        let token_lp = &log_probs_v[i * n..(i + 1) * n];
        let mut ranked: Vec<(usize, f32)> = token_lp.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        let argmax_id = ranked[0].0;
        println!(
            "{:>3} tok={:<7} bytes={}..{} text={:?} viterbi={}({}) argmax={}({:.2})",
            i,
            tid,
            bs,
            be,
            token_text,
            engine.label_info.id2label[label_id],
            label_id,
            engine.label_info.id2label[argmax_id],
            ranked[0].1.exp(),
        );
    }
    Ok(ExitCode::SUCCESS)
}

fn emit_and_exit(mode: Mode, text: &str, spans: &[DetectedSpan]) -> Result<ExitCode> {
    match mode {
        Mode::Check => {
            if spans.is_empty() {
                Ok(ExitCode::SUCCESS)
            } else {
                for s in spans {
                    let (line, col) = byte_to_line_col(text, s.byte_start);
                    eprintln!(
                        "{}:{}: {} ({}..{}) {:?}",
                        line, col, s.label, s.byte_start, s.byte_end, s.text
                    );
                }
                Ok(ExitCode::from(1))
            }
        }
        Mode::Redact => {
            print!("{}", redact(text, spans));
            Ok(ExitCode::SUCCESS)
        }
        Mode::List => {
            let json: Vec<serde_json::Value> = spans
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "label": s.label,
                        "byte_start": s.byte_start,
                        "byte_end": s.byte_end,
                        "text": s.text,
                    })
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&json)?);
            Ok(ExitCode::SUCCESS)
        }
        Mode::Debug | Mode::Stream => unreachable!("handled above"),
    }
}

fn read_input(cli: &Cli) -> Result<String> {
    if let Some(t) = &cli.text {
        return Ok(t.clone());
    }
    if let Some(p) = &cli.file {
        return std::fs::read_to_string(p).with_context(|| format!("read {}", p.display()));
    }
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf)?;
    Ok(buf)
}

fn argmax_decode(log_probs: &[f32], seq_len: usize, n: usize) -> Vec<usize> {
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

fn pick_device(use_metal: bool) -> Result<Device> {
    if use_metal {
        return Ok(Device::new_metal(0)?);
    }
    Ok(Device::Cpu)
}

fn byte_to_line_col(text: &str, byte_offset: usize) -> (usize, usize) {
    let mut line = 1usize;
    let mut col = 1usize;
    for (i, b) in text.bytes().enumerate() {
        if i >= byte_offset {
            break;
        }
        if b == b'\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}
