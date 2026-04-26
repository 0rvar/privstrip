use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f64,
    pub factor: f64,
    pub original_max_position_embeddings: usize,
    pub beta_fast: f64,
    pub beta_slow: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    pub sliding_window: usize,
    pub rms_norm_eps: f64,
    pub rope_parameters: RopeParameters,
    #[serde(default = "default_swiglu_limit")]
    pub swiglu_limit: f32,
}

fn default_swiglu_limit() -> f32 {
    7.0
}

impl ModelConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("read {}", path.display()))?;
        Ok(serde_json::from_str(&raw)?)
    }

    pub fn num_classes(&self) -> usize {
        33
    }
}
