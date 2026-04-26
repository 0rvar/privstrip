use anyhow::{Context, Result};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module, RmsNorm, VarBuilder, ops};

use crate::config::ModelConfig;

pub struct Transformer {
    pub embedding: Tensor,
    pub blocks: Vec<TransformerBlock>,
    pub final_norm: RmsNorm,
    pub unembedding: Tensor,
    pub config: ModelConfig,
    pub device: Device,
    pub dtype: DType,
}

pub struct TransformerBlock {
    pub attn: AttentionBlock,
    pub mlp: MLPBlock,
}

pub struct AttentionBlock {
    pub norm: RmsNorm,
    pub qkv: Linear,
    pub out: Linear,
    pub sinks: Tensor, // [num_attention_heads], stored in log2 units
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub sliding_window: usize,
    pub rope_cos: Tensor, // [max_pos, head_dim/2]
    pub rope_sin: Tensor,
}

pub struct MLPBlock {
    pub norm: RmsNorm,
    pub gate: Linear,
    pub mlp1_weight: Tensor, // [num_experts, hidden, 2*intermediate] (gpt-oss layout: x @ W)
    pub mlp1_bias: Tensor,   // [num_experts, 2*intermediate]
    pub mlp2_weight: Tensor, // [num_experts, intermediate, hidden]
    pub mlp2_bias: Tensor,   // [num_experts, hidden]
    pub num_experts: usize,
    pub experts_per_tok: usize,
    pub swiglu_limit: f32,
}

impl Transformer {
    pub fn load(
        weights_path: &std::path::Path,
        config: ModelConfig,
        device: Device,
    ) -> Result<Self> {
        // Weights on disk are bf16; we run the forward in f32 for numerical headroom and op coverage.
        let dtype = DType::F32;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device)
                .with_context(|| format!("load {}", weights_path.display()))?
        };

        let embedding = vb.get((config.vocab_size, config.hidden_size), "embedding.weight")?;
        let final_scale = vb.get(config.hidden_size, "norm.scale")?;
        let final_norm = RmsNorm::new(final_scale, config.rms_norm_eps);
        let unembedding =
            vb.get((config.num_classes(), config.hidden_size), "unembedding.weight")?;

        let (rope_cos, rope_sin) = build_yarn_rope(&config, &device, dtype)?;

        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let block_vb = vb.pp(format!("block.{i}"));
            let attn = AttentionBlock::load(
                &block_vb.pp("attn"),
                &config,
                rope_cos.clone(),
                rope_sin.clone(),
            )?;
            let mlp = MLPBlock::load(&block_vb.pp("mlp"), &config)?;
            blocks.push(TransformerBlock { attn, mlp });
        }

        Ok(Self {
            embedding,
            blocks,
            final_norm,
            unembedding,
            config,
            device,
            dtype,
        })
    }

    /// Forward pass. `tokens` is a 1-D u32 Tensor of length T. Returns logits [T, num_classes] in f32.
    pub fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let mut x = self.embedding.index_select(tokens, 0)?; // [T, hidden]
        for block in &self.blocks {
            x = block.attn.forward(&x)?;
            x = block.mlp.forward(&x)?;
        }
        let x = self.final_norm.forward(&x)?;
        let logits = x.matmul(&self.unembedding.t()?.contiguous()?)?; // [T, num_classes]
        Ok(logits)
    }
}

impl AttentionBlock {
    fn load(
        vb: &VarBuilder,
        config: &ModelConfig,
        rope_cos: Tensor,
        rope_sin: Tensor,
    ) -> Result<Self> {
        let h = config.hidden_size;
        let nq = config.num_attention_heads;
        let nkv = config.num_key_value_heads;
        let d = config.head_dim;
        let qkv_dim = (nq + 2 * nkv) * d;

        let norm_scale = vb.pp("norm").get(h, "scale")?;
        let norm = RmsNorm::new(norm_scale, config.rms_norm_eps);

        let qkv_w = vb.pp("qkv").get((qkv_dim, h), "weight")?;
        let qkv_b = vb.pp("qkv").get(qkv_dim, "bias")?;
        let qkv = Linear::new(qkv_w, Some(qkv_b));

        let out_w = vb.pp("out").get((h, nq * d), "weight")?;
        let out_b = vb.pp("out").get(h, "bias")?;
        let out = Linear::new(out_w, Some(out_b));

        let sinks = vb.get(nq, "sinks")?; // f32 in checkpoint, loaded as configured dtype

        Ok(Self {
            norm,
            qkv,
            out,
            sinks,
            num_q_heads: nq,
            num_kv_heads: nkv,
            head_dim: d,
            sliding_window: config.sliding_window,
            rope_cos,
            rope_sin,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let t = x.dim(0)?;
        let device = x.device();
        let dtype = x.dtype();

        let normed = self.norm.forward(x)?;
        let qkv = self.qkv.forward(&normed)?; // [T, qkv_dim]

        let nq = self.num_q_heads;
        let nkv = self.num_kv_heads;
        let d = self.head_dim;

        let q = qkv.narrow(1, 0, nq * d)?;
        let k = qkv.narrow(1, nq * d, nkv * d)?;
        let v = qkv.narrow(1, (nq + nkv) * d, nkv * d)?;

        // Reshape into (B=1, H, T, D) for rope_i.
        let q = q
            .reshape((t, nq, d))?
            .transpose(0, 1)?
            .unsqueeze(0)?
            .contiguous()?;
        let k = k
            .reshape((t, nkv, d))?
            .transpose(0, 1)?
            .unsqueeze(0)?
            .contiguous()?;
        let v = v
            .reshape((t, nkv, d))?
            .transpose(0, 1)?
            .unsqueeze(0)?
            .contiguous()?;

        let cos = self.rope_cos.narrow(0, 0, t)?.contiguous()?;
        let sin = self.rope_sin.narrow(0, 0, t)?.contiguous()?;

        let q = candle_nn::rotary_emb::rope_i(&q, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope_i(&k, &cos, &sin)?;

        // qk_scale split between q and k: each scaled by d^(-1/4) so q·k is scaled by d^(-1/2).
        let qk_scale = (d as f64).powf(-0.25);
        let q = q.affine(qk_scale, 0.0)?;
        let k = k.affine(qk_scale, 0.0)?;

        let q = q.squeeze(0)?; // [nq, T, d]
        let k = k.squeeze(0)?; // [nkv, T, d]
        let v = v.squeeze(0)?; // [nkv, T, d]

        let q_mult = nq / nkv;
        let k = repeat_kv(&k, q_mult)?; // [nq, T, d]
        let v = repeat_kv(&v, q_mult)?; // [nq, T, d]

        // scores: q @ k.T -> [nq, T, T]
        let kt = k.transpose(1, 2)?.contiguous()?;
        let mut scores = q.matmul(&kt)?;

        // Causal sliding-window mask.
        let mask = causal_sliding_mask(t, self.sliding_window, device, dtype)?; // [T, T]
        scores = scores.broadcast_add(&mask.unsqueeze(0)?)?; // [nq, T, T]

        // Attention sinks: append one virtual key per head with logit = sink * ln(2).
        let ln2 = std::f64::consts::LN_2;
        let sink_per_head = self.sinks.affine(ln2, 0.0)?.to_dtype(dtype)?; // [nq]
        let sink_col = sink_per_head
            .reshape((nq, 1, 1))?
            .broadcast_as((nq, t, 1))?
            .contiguous()?; // [nq, T, 1]
        let scores_with_sink = Tensor::cat(&[&scores, &sink_col], D::Minus1)?; // [nq, T, T+1]

        let weights = ops::softmax(&scores_with_sink, D::Minus1)?;
        let weights = weights.narrow(D::Minus1, 0, t)?.contiguous()?; // [nq, T, T]

        let attn_out = weights.matmul(&v)?; // [nq, T, d]
        let attn_out = attn_out
            .transpose(0, 1)?
            .contiguous()?
            .reshape((t, nq * d))?;

        let projected = self.out.forward(&attn_out)?; // [T, hidden]
        x.add(&projected).map_err(Into::into)
    }
}

impl MLPBlock {
    fn load(vb: &VarBuilder, config: &ModelConfig) -> Result<Self> {
        let h = config.hidden_size;
        let inter = config.intermediate_size;
        let ne = config.num_local_experts;

        let norm_scale = vb.pp("norm").get(h, "scale")?;
        let norm = RmsNorm::new(norm_scale, config.rms_norm_eps);

        let gate_w = vb.pp("gate").get((ne, h), "weight")?;
        let gate_b = vb.pp("gate").get(ne, "bias")?;
        let gate = Linear::new(gate_w, Some(gate_b));

        // gpt-oss expert layout: x @ W (W is [in, out], not the standard PyTorch [out, in]).
        let mlp1_weight = vb.pp("swiglu").get((ne, h, 2 * inter), "weight")?;
        let mlp1_bias = vb.pp("swiglu").get((ne, 2 * inter), "bias")?;
        let mlp2_weight = vb.pp("out").get((ne, inter, h), "weight")?;
        let mlp2_bias = vb.pp("out").get((ne, h), "bias")?;

        Ok(Self {
            norm,
            gate,
            mlp1_weight,
            mlp1_bias,
            mlp2_weight,
            mlp2_bias,
            num_experts: ne,
            experts_per_tok: config.num_experts_per_tok,
            swiglu_limit: config.swiglu_limit,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let t = x.dim(0)?;
        let h = x.dim(1)?;
        let device = x.device();
        let dtype = x.dtype();

        let normed = self.norm.forward(x)?;
        let gate_logits = self.gate.forward(&normed)?; // [T, num_experts]

        let gate_logits_f32 = gate_logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let k = self.experts_per_tok;
        let mut indices_flat: Vec<u32> = Vec::with_capacity(t * k);
        let mut weights_flat: Vec<f32> = Vec::with_capacity(t * k);
        for token_logits in &gate_logits_f32 {
            let (top_idx, top_logits) = top_k(token_logits, k);
            let max = top_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = top_logits.iter().map(|l| (l - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for j in 0..k {
                indices_flat.push(top_idx[j] as u32);
                weights_flat.push(exps[j] / sum);
            }
        }

        let mut per_expert: Vec<Vec<(u32, f32)>> = vec![Vec::new(); self.num_experts];
        for token in 0..t {
            for j in 0..k {
                let e = indices_flat[token * k + j] as usize;
                per_expert[e].push((token as u32, weights_flat[token * k + j]));
            }
        }

        let mut out = Tensor::zeros((t, h), dtype, device)?;
        for (e, assigns) in per_expert.iter().enumerate() {
            if assigns.is_empty() {
                continue;
            }
            let n_e = assigns.len();
            let token_idx_v: Vec<u32> = assigns.iter().map(|(tok, _)| *tok).collect();
            let weights_v: Vec<f32> = assigns.iter().map(|(_, w)| *w).collect();
            let token_idx = Tensor::from_vec(token_idx_v, n_e, device)?;
            let weights_col = Tensor::from_vec(weights_v, (n_e, 1), device)?.to_dtype(dtype)?;

            let x_e = normed.index_select(&token_idx, 0)?; // [n_e, hidden]

            let w1 = self.mlp1_weight.i(e)?.contiguous()?; // [hidden, 2*intermediate]
            let b1 = self.mlp1_bias.i(e)?; // [2*intermediate]
            let h1 = x_e.matmul(&w1)?.broadcast_add(&b1)?;
            let h1 = swiglu(&h1, self.swiglu_limit)?; // [n_e, intermediate]

            let w2 = self.mlp2_weight.i(e)?.contiguous()?; // [intermediate, hidden]
            let b2 = self.mlp2_bias.i(e)?; // [hidden]
            let h2 = h1.matmul(&w2)?.broadcast_add(&b2)?;

            let weighted = h2.broadcast_mul(&weights_col)?; // [n_e, hidden]
            out = out.index_add(&token_idx, &weighted, 0)?;
        }

        x.add(&out).map_err(Into::into)
    }
}

fn top_k(values: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    let mut paired: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    if k < paired.len() {
        paired.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
        paired.truncate(k);
    }
    paired.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let idx = paired.iter().map(|(i, _)| *i).collect();
    let val = paired.iter().map(|(_, v)| *v).collect();
    (idx, val)
}

/// SwiGLU as used by openai/privacy-filter (gpt-oss): GELU-style activation on the GLU half
/// (sigmoid(1.702 * x)) plus a +1 bias on the linear half, with asymmetric clamping.
fn swiglu(x: &Tensor, limit: f32) -> Result<Tensor> {
    let last = x.dim(D::Minus1)?;
    let half = last / 2;
    let x_glu = x.narrow(D::Minus1, 0, half)?;
    let x_lin = x.narrow(D::Minus1, half, half)?;

    let x_glu = x_glu.minimum(limit as f64)?;
    let x_lin = x_lin.clamp(-(limit as f64), limit as f64)?;

    let alpha = 1.702f64;
    let scaled = x_glu.affine(alpha, 0.0)?;
    let sig = ops::sigmoid(&scaled)?;
    let activated = (&x_glu * &sig)?;
    let lin_plus_one = x_lin.affine(1.0, 1.0)?;
    let out = (&activated * &lin_plus_one)?;
    Ok(out)
}

fn repeat_kv(t: &Tensor, q_mult: usize) -> Result<Tensor> {
    if q_mult == 1 {
        return Ok(t.clone());
    }
    let (nkv, seq, d) = t.dims3()?;
    let expanded = t
        .unsqueeze(1)?
        .expand((nkv, q_mult, seq, d))?
        .contiguous()?;
    expanded.reshape((nkv * q_mult, seq, d)).map_err(Into::into)
}

fn causal_sliding_mask(t: usize, window: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let neg_inf = f32::NEG_INFINITY;
    let mut data = vec![0f32; t * t];
    for i in 0..t {
        let lo = i.saturating_sub(window);
        for j in 0..t {
            if j < lo || j > i {
                data[i * t + j] = neg_inf;
            }
        }
    }
    Tensor::from_vec(data, (t, t), device)?
        .to_dtype(dtype)
        .map_err(Into::into)
}

fn build_yarn_rope(
    config: &ModelConfig,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let d = config.head_dim;
    let d_half = d / 2;
    let base = config.rope_parameters.rope_theta;
    let scaling_factor = config.rope_parameters.factor;
    let initial_ctx = config.rope_parameters.original_max_position_embeddings as f64;
    let ntk_alpha = config.rope_parameters.beta_slow;
    let ntk_beta = config.rope_parameters.beta_fast;
    // Cap the precomputed table at a generous but bounded length.
    let max_pos = ((initial_ctx * scaling_factor) as usize).min(8192);

    let freq: Vec<f64> = (0..d_half)
        .map(|i| base.powf((2.0 * i as f64) / d as f64))
        .collect();

    let (concentration, inv_freq): (f64, Vec<f64>) = if scaling_factor > 1.0 {
        let concentration = 0.1 * scaling_factor.ln() + 1.0;
        let dh = d_half as f64;
        let two_pi = std::f64::consts::TAU;
        let low = dh * (initial_ctx / (ntk_beta * two_pi)).ln() / base.ln();
        let high = dh * (initial_ctx / (ntk_alpha * two_pi)).ln() / base.ln();
        anyhow::ensure!(
            0.0 < low && low < high && high < dh - 1.0,
            "YaRN ntk-by-parts boundary check failed: low={low}, high={high}, d_half={dh}"
        );
        let mut inv = Vec::with_capacity(d_half);
        for i in 0..d_half {
            let interpolation = 1.0 / (scaling_factor * freq[i]);
            let extrapolation = 1.0 / freq[i];
            let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
            let mask = 1.0 - ramp;
            inv.push(interpolation * (1.0 - mask) + extrapolation * mask);
        }
        (concentration, inv)
    } else {
        (1.0, freq.iter().map(|f| 1.0 / f).collect())
    };

    let mut cos = vec![0f32; max_pos * d_half];
    let mut sin = vec![0f32; max_pos * d_half];
    for t in 0..max_pos {
        for i in 0..d_half {
            let theta = (t as f64) * inv_freq[i];
            cos[t * d_half + i] = (theta.cos() * concentration) as f32;
            sin[t * d_half + i] = (theta.sin() * concentration) as f32;
        }
    }
    let cos_t = Tensor::from_vec(cos, (max_pos, d_half), device)?.to_dtype(dtype)?;
    let sin_t = Tensor::from_vec(sin, (max_pos, d_half), device)?.to_dtype(dtype)?;
    Ok((cos_t, sin_t))
}
