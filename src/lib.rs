//! PII detection with the `openai/privacy-filter` model family, running the
//! weights directly through candle — no Python, no ONNX runtime.
//!
//! The crate ships both this library and the `privstrip` CLI. The library is the
//! inference engine and nothing more: it loads a checkpoint from a directory and
//! turns text into byte-offset spans. Anything deployment-shaped — HTTP serving,
//! fetching weights from object storage, containers — lives in the Timely infra
//! repo at `services/privstrip`, which depends on this crate.
//!
//! ```no_run
//! use privstrip::{DecoderMode, Engine, DEFAULT_OPERATING_POINT, pick_device};
//!
//! let engine = Engine::load(std::path::Path::new("models/base"), pick_device(false)?)?;
//! let result = engine.detect(
//!     "Call John Smith at 555-1234",
//!     DecoderMode::Viterbi,
//!     DEFAULT_OPERATING_POINT,
//! )?;
//! for span in &result.spans {
//!     println!("{} {}..{}", span.label, span.byte_start, span.byte_end);
//! }
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! Span offsets are byte offsets into the input text, not char indices.

pub mod config;
pub mod labels;
pub mod model;
pub mod spans;
pub mod timing;
pub mod viterbi;

mod engine;

pub use engine::{
    DEFAULT_OPERATING_POINT, DecoderMode, DetectionResult, Engine, argmax_decode,
    detection_error_json, detection_json, pick_device,
};
pub use spans::{DetectedSpan, extract_spans, redact};

/// Re-exported so consumers can name the device an `Engine` runs on without
/// taking their own candle dependency (and risking a version mismatch).
pub use candle_core::Device;
