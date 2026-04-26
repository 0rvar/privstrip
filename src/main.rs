mod config;
mod labels;
mod model;
mod spans;
mod viterbi;

use std::io::Read;
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
enum Mode {
    /// Print PII locations and exit 1 if any are found.
    Check,
    /// Print the input with PII replaced by <LABEL> placeholders.
    Redact,
    /// Print a JSON list of detected PII spans.
    List,
    /// Print per-token predictions for debugging.
    Debug,
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
    #[arg(short = 'm', long, default_value = ".")]
    model_dir: PathBuf,
    /// Force CPU even if Metal is available.
    #[arg(long)]
    cpu: bool,
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

    let text = read_input(&cli)?;
    if text.is_empty() {
        // Empty input: nothing to detect; emit-and-exit per mode.
        return emit_and_exit(cli.mode, &text, &[]);
    }

    let device = pick_device(cli.cpu)?;

    let config_path = cli.model_dir.join("config.json");
    let tokenizer_path = cli.model_dir.join("tokenizer.json");
    let weights_path = cli.model_dir.join("model.safetensors");

    let config = ModelConfig::from_file(&config_path)?;
    let label_info = LabelInfo::from_config(&config_path)?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;

    let encoding = tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
    let token_ids = encoding.get_ids().to_vec();
    let offsets: Vec<(usize, usize)> = encoding.get_offsets().to_vec();

    if token_ids.is_empty() {
        return emit_and_exit(cli.mode, &text, &[]);
    }

    let model = Transformer::load(&weights_path, config, device.clone())?;

    let tokens = Tensor::from_vec(token_ids.clone(), token_ids.len(), &device)?;
    let logits = model.forward(&tokens)?; // [T, num_classes]
    let log_probs = ops::log_softmax(&logits.to_dtype(DType::F32)?, D::Minus1)?;
    let log_probs_v: Vec<f32> = log_probs.flatten_all()?.to_vec1()?;

    let decoder = ViterbiDecoder::new(&label_info);
    let label_path = decoder.decode(&log_probs_v, token_ids.len());

    if matches!(cli.mode, Mode::Debug) {
        let n = label_info.num_classes();
        for (i, &tid) in token_ids.iter().enumerate() {
            let (bs, be) = offsets[i];
            let token_text = text.get(bs..be).unwrap_or("");
            let label_id = label_path[i];
            // Top-3 raw probabilities for context.
            let token_lp = &log_probs_v[i * n..(i + 1) * n];
            let mut ranked: Vec<(usize, f32)> =
                token_lp.iter().copied().enumerate().collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let argmax_id = ranked[0].0;
            println!(
                "{:>3} tok={:<7} bytes={}..{} text={:?} viterbi={}({}) argmax={}({:.2})",
                i,
                tid,
                bs,
                be,
                token_text,
                label_info.id2label[label_id],
                label_id,
                label_info.id2label[argmax_id],
                ranked[0].1.exp(),
            );
        }
        return Ok(ExitCode::SUCCESS);
    }

    let spans = extract_spans(&label_path, &offsets, &text, &label_info);
    emit_and_exit(cli.mode, &text, &spans)
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
        Mode::Debug => unreachable!("debug handled above"),
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

fn pick_device(force_cpu: bool) -> Result<Device> {
    if force_cpu {
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "metal")]
    {
        if let Ok(d) = Device::new_metal(0) {
            return Ok(d);
        }
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
