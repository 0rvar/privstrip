use std::io::{BufRead, Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use candle_core::{D, DType, Tensor};
use candle_nn::ops;
use clap::{Parser, ValueEnum};

use privstrip::spans::{DetectedSpan, redact};
use privstrip::timing::{Stage, time_stage};
use privstrip::{
    DEFAULT_OPERATING_POINT, DecoderMode, Engine, argmax_decode, detection_error_json,
    detection_json, pick_device,
};

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
    #[arg(short = 'm', long, default_value = "models/base")]
    model_dir: PathBuf,
    /// Use Metal (Apple GPU) instead of CPU. Off by default because some sandboxes can't init Metal.
    #[arg(long)]
    metal: bool,
    /// Decoding strategy applied to per-token logits.
    #[arg(long, value_enum, default_value_t = DecoderMode::Viterbi)]
    decoder: DecoderMode,
    /// Named operating point in `viterbi_calibration.json`. Only consulted in
    /// viterbi mode. The shipped checkpoint has all-zero biases at the default
    /// operating point, so the default is mathematically a no-op.
    #[arg(long, default_value = DEFAULT_OPERATING_POINT)]
    operating_point: String,
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
    let engine = Engine::load(&cli.model_dir, device)?;
    // Reject an unknown operating point before reading input, so a typo fails
    // immediately rather than once per row of a long stream. Checked in argmax
    // mode too, where the value is unused, so that a typo is never silently
    // accepted depending on the decoder.
    engine.decoder_for(&cli.operating_point)?;

    match cli.mode {
        Mode::Stream => run_stream(&engine, cli.decoder, &cli.operating_point),
        Mode::Debug => {
            let text = read_input(&cli)?;
            run_debug(&engine, &text, cli.decoder, &cli.operating_point)
        }
        mode => {
            let text = read_input(&cli)?;
            let result = engine.detect(&text, cli.decoder, &cli.operating_point)?;
            emit_and_exit(mode, &text, &result.spans)
        }
    }
}

fn run_stream(
    engine: &Engine,
    decoder_mode: DecoderMode,
    operating_point: &str,
) -> Result<ExitCode> {
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
        let response = process_stream_line(engine, trimmed, decoder_mode, operating_point);
        time_stage(Stage::Serialize, || -> Result<()> {
            serde_json::to_writer(&mut out, &response)?;
            out.write_all(b"\n")?;
            out.flush()?;
            Ok(())
        })?;
    }
    privstrip::timing::report(&mut std::io::stderr())?;
    Ok(ExitCode::SUCCESS)
}

fn process_stream_line(
    engine: &Engine,
    line: &str,
    decoder_mode: DecoderMode,
    operating_point: &str,
) -> serde_json::Value {
    #[derive(serde::Deserialize)]
    struct In {
        id: serde_json::Value,
        text: String,
    }
    let parsed: Result<In, _> = serde_json::from_str(line);
    let (id, text) = match parsed {
        Ok(v) => (v.id, v.text),
        Err(e) => {
            return detection_error_json(
                serde_json::Value::Null,
                format!("invalid input json: {e}"),
            );
        }
    };
    let started = std::time::Instant::now();
    match engine.detect(&text, decoder_mode, operating_point) {
        Ok(res) => detection_json(id, &res, started.elapsed().as_micros() as u64),
        Err(e) => detection_error_json(id, format!("{e:#}")),
    }
}

fn run_debug(
    engine: &Engine,
    text: &str,
    decoder_mode: DecoderMode,
    operating_point: &str,
) -> Result<ExitCode> {
    if text.is_empty() {
        return Ok(ExitCode::SUCCESS);
    }
    let encoding = engine
        .tokenizer()
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
    let token_ids = encoding.get_ids().to_vec();
    let offsets: Vec<(usize, usize)> = encoding.get_offsets().to_vec();
    if token_ids.is_empty() {
        return Ok(ExitCode::SUCCESS);
    }

    let tokens = Tensor::from_vec(token_ids.clone(), token_ids.len(), engine.device())?;
    let logits = engine.model().forward(&tokens)?;
    let log_probs_t = ops::log_softmax(&logits.to_dtype(DType::F32)?, D::Minus1)?;
    let log_probs_v: Vec<f32> = log_probs_t.flatten_all()?.to_vec1()?;
    let n = engine.label_info().num_classes();
    let label_path = match decoder_mode {
        DecoderMode::Viterbi => engine
            .decoder_for(operating_point)?
            .decode(&log_probs_v, token_ids.len()),
        DecoderMode::Argmax => argmax_decode(&log_probs_v, token_ids.len(), n),
    };
    let dump_top_k = std::env::var("PRIVSTRIP_DEBUG_TOPK")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
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
            engine.label_info().id2label[label_id],
            label_id,
            engine.label_info().id2label[argmax_id],
            ranked[0].1.exp(),
        );
        if dump_top_k > 0 {
            let top: Vec<String> = ranked
                .iter()
                .take(dump_top_k)
                .map(|(id, lp)| format!("{}={:+.6}", engine.label_info().id2label[*id], lp))
                .collect();
            println!("    top: {}", top.join(" "));
        }
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
