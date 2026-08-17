use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::{Context, Result};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module, RmsNorm, VarBuilder, ops};

use crate::config::ModelConfig;
use crate::timing::{Stage, time_stage};

/// Tensor naming convention on disk.
///
/// `Custom` is the layout shipped by `openai/privacy-filter` itself: a fused
/// `block.{i}.attn.qkv` projection, scale-named norms, no classifier bias.
/// `Hf` is the standard `transformers` MoE serialization used by community
/// fine-tunes such as `OpenMed/privacy-filter-multilingual`: separate
/// `q_proj`/`k_proj`/`v_proj`, weight-named norms, and a `score` head with bias.
/// The math is identical; only the on-disk packing differs.
#[derive(Debug, Clone, Copy)]
enum Naming {
    Custom,
    Hf,
}

/// Memory the attention-mask cache may retain before it is dropped wholesale.
/// See `Transformer::cached_mask`.
const MASK_CACHE_MAX_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// Bytes a dense `[T, T]` f32 sliding-window mask occupies.
fn mask_bytes(t: usize) -> usize {
    t.saturating_mul(t).saturating_mul(4)
}

fn cache_bytes(cache: &HashMap<usize, Tensor>) -> usize {
    cache.keys().copied().map(mask_bytes).sum()
}

pub struct Transformer {
    pub embedding: Tensor,
    pub blocks: Vec<TransformerBlock>,
    pub final_norm: RmsNorm,
    pub sliding_window: usize,
    // Bidirectional sliding-window mask depends only on (T, sliding_window,
    // device, dtype). The window/device/dtype are fixed per Transformer
    // instance, so caching by T avoids rebuilding the same TxT tensor on every
    // forward pass. Cap is bounded by the RoPE table size (<= 8192).
    mask_cache: Mutex<HashMap<usize, Tensor>>,
    // Materialized transpose of the unembedding matrix. matmul needs the
    // contiguous form; doing the transpose+contiguous once at load-time saves
    // the copy on every forward.
    unembedding_t: Tensor,
    // Optional classifier-head bias. The base checkpoint's `unembedding` is a
    // bare matmul; HF-naming checkpoints (e.g. OpenMed multilingual) add a
    // `score.bias` from `nn.Linear`.
    unembedding_b: Option<Tensor>,
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

        let naming = if vb.contains_tensor("embedding.weight") {
            Naming::Custom
        } else if vb.contains_tensor("model.embed_tokens.weight") {
            Naming::Hf
        } else {
            anyhow::bail!(
                "could not detect tensor naming convention: neither `embedding.weight` \
                 (openai/privacy-filter) nor `model.embed_tokens.weight` (HF MoE) found"
            );
        };

        let (embedding, final_scale, unembedding, unembedding_b) = match naming {
            Naming::Custom => {
                let e = vb.get((config.vocab_size, config.hidden_size), "embedding.weight")?;
                let n = vb.get(config.hidden_size, "norm.scale")?;
                let u = vb.get((config.num_classes(), config.hidden_size), "unembedding.weight")?;
                (e, n, u, None)
            }
            Naming::Hf => {
                let e = vb.get((config.vocab_size, config.hidden_size), "model.embed_tokens.weight")?;
                let n = vb.get(config.hidden_size, "model.norm.weight")?;
                let u = vb.get((config.num_classes(), config.hidden_size), "score.weight")?;
                let b = vb.get(config.num_classes(), "score.bias")?;
                (e, n, u, Some(b))
            }
        };
        let final_norm = RmsNorm::new(final_scale, config.rms_norm_eps);

        let (rope_cos, rope_sin) = build_yarn_rope(&config, &device, dtype)?;

        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        let sliding_window = config.sliding_window;
        for i in 0..config.num_hidden_layers {
            let (block_vb, attn_prefix, mlp_prefix) = match naming {
                Naming::Custom => (vb.pp(format!("block.{i}")), "attn", "mlp"),
                Naming::Hf => (vb.pp(format!("model.layers.{i}")), "self_attn", "mlp"),
            };
            let attn = AttentionBlock::load(
                &block_vb.pp(attn_prefix),
                &config,
                rope_cos.clone(),
                rope_sin.clone(),
                naming,
                &block_vb,
            )?;
            let mlp = MLPBlock::load(&block_vb.pp(mlp_prefix), &config, naming, &block_vb)?;
            blocks.push(TransformerBlock { attn, mlp });
        }

        let unembedding_t = unembedding.t()?.contiguous()?;
        Ok(Self {
            embedding,
            blocks,
            final_norm,
            unembedding_t,
            unembedding_b,
            sliding_window,
            mask_cache: Mutex::new(HashMap::new()),
        })
    }

    fn cached_mask(&self, t: usize, device: &Device, dtype: DType) -> Result<Tensor> {
        {
            let cache = self.mask_cache.lock().unwrap();
            if let Some(m) = cache.get(&t) {
                return Ok(m.clone());
            }
        }
        let m = bidirectional_sliding_mask(t, self.sliding_window, device, dtype)?;
        let mut cache = self.mask_cache.lock().unwrap();
        // An entry costs 4*T^2 bytes — 1 GB at the T=16384 ceiling — so the
        // budget has to be in bytes, not entries: 64 long inputs would be tens
        // of GB. A long-lived server sees an unbounded spread of token counts,
        // so drop the whole cache once the retained masks exceed the budget.
        // Correctness never depends on a hit: the mask is a pure function of
        // (T, window, device, dtype).
        if cache_bytes(&cache) + mask_bytes(t) > MASK_CACHE_MAX_BYTES {
            cache.clear();
        }
        // A mask too large to ever coexist with the budget is handed back
        // uncached rather than evicting everything on every long request.
        if mask_bytes(t) > MASK_CACHE_MAX_BYTES {
            return Ok(m);
        }
        Ok(cache.entry(t).or_insert(m).clone())
    }

    /// Forward pass. `tokens` is a 1-D u32 Tensor of length T. Returns logits [T, num_classes] in f32.
    pub fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let t = tokens.dim(0)?;
        let max_pos = self.rope_cos_dim0()?;
        anyhow::ensure!(
            t <= max_pos,
            "input is {t} tokens but RoPE table is sized for {max_pos}; \
             raise the cap in build_yarn_rope or chunk the input"
        );

        let device = tokens.device().clone();
        let dtype = self.embedding.dtype();
        let mask = self.cached_mask(t, &device, dtype)?;

        let mut x = self.embedding.index_select(tokens, 0)?; // [T, hidden]
        for block in &self.blocks {
            x = time_stage(Stage::ForwardAttn, || block.attn.forward(&x, &mask))?;
            x = block.mlp.forward(&x)?;
        }
        let x = self.final_norm.forward(&x)?;
        let mut logits = x.matmul(&self.unembedding_t)?; // [T, num_classes]
        if let Some(b) = &self.unembedding_b {
            logits = logits.broadcast_add(b)?;
        }
        Ok(logits)
    }

    fn rope_cos_dim0(&self) -> Result<usize> {
        // The RoPE table lives on each AttentionBlock; they're all the same length.
        Ok(self.blocks[0].attn.rope_cos.dim(0)?)
    }
}

impl AttentionBlock {
    fn load(
        vb: &VarBuilder,
        config: &ModelConfig,
        rope_cos: Tensor,
        rope_sin: Tensor,
        naming: Naming,
        block_vb: &VarBuilder,
    ) -> Result<Self> {
        let h = config.hidden_size;
        let nq = config.num_attention_heads;
        let nkv = config.num_key_value_heads;
        let d = config.head_dim;
        let qkv_dim = (nq + 2 * nkv) * d;

        let (norm_scale, qkv_w, qkv_b, out_w, out_b, sinks) = match naming {
            Naming::Custom => {
                let n = vb.pp("norm").get(h, "scale")?;
                let qkv_w = vb.pp("qkv").get((qkv_dim, h), "weight")?;
                let qkv_b = vb.pp("qkv").get(qkv_dim, "bias")?;
                let out_w = vb.pp("out").get((h, nq * d), "weight")?;
                let out_b = vb.pp("out").get(h, "bias")?;
                let sinks = vb.get(nq, "sinks")?;
                (n, qkv_w, qkv_b, out_w, out_b, sinks)
            }
            Naming::Hf => {
                // input_layernorm sits at block-level, not inside self_attn.
                let n = block_vb.pp("input_layernorm").get(h, "weight")?;
                // Concatenate q/k/v along dim 0 to match the fused layout the
                // forward pass slices with narrow().
                let q_w = vb.pp("q_proj").get((nq * d, h), "weight")?;
                let k_w = vb.pp("k_proj").get((nkv * d, h), "weight")?;
                let v_w = vb.pp("v_proj").get((nkv * d, h), "weight")?;
                let qkv_w = Tensor::cat(&[&q_w, &k_w, &v_w], 0)?.contiguous()?;
                let q_b = vb.pp("q_proj").get(nq * d, "bias")?;
                let k_b = vb.pp("k_proj").get(nkv * d, "bias")?;
                let v_b = vb.pp("v_proj").get(nkv * d, "bias")?;
                let qkv_b = Tensor::cat(&[&q_b, &k_b, &v_b], 0)?.contiguous()?;
                let out_w = vb.pp("o_proj").get((h, nq * d), "weight")?;
                let out_b = vb.pp("o_proj").get(h, "bias")?;
                let sinks = vb.get(nq, "sinks")?;
                (n, qkv_w, qkv_b, out_w, out_b, sinks)
            }
        };

        let norm = RmsNorm::new(norm_scale, config.rms_norm_eps);
        let qkv = Linear::new(qkv_w, Some(qkv_b));
        let out = Linear::new(out_w, Some(out_b));

        Ok(Self {
            norm,
            qkv,
            out,
            sinks,
            num_q_heads: nq,
            num_kv_heads: nkv,
            head_dim: d,
            rope_cos,
            rope_sin,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let t = x.dim(0)?;
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

        // Bidirectional sliding-window mask is shared across blocks — built once per forward.
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
    fn load(
        vb: &VarBuilder,
        config: &ModelConfig,
        naming: Naming,
        block_vb: &VarBuilder,
    ) -> Result<Self> {
        let h = config.hidden_size;
        let inter = config.intermediate_size;
        let ne = config.num_local_experts;

        let (norm_scale, gate_w, gate_b, mlp1_weight, mlp1_bias, mlp2_weight, mlp2_bias) =
            match naming {
                Naming::Custom => {
                    let n = vb.pp("norm").get(h, "scale")?;
                    let gw = vb.pp("gate").get((ne, h), "weight")?;
                    let gb = vb.pp("gate").get(ne, "bias")?;
                    // gpt-oss expert layout: x @ W (W is [in, out], not the standard PyTorch [out, in]).
                    let m1w = vb.pp("swiglu").get((ne, h, 2 * inter), "weight")?;
                    let m1b = vb.pp("swiglu").get((ne, 2 * inter), "bias")?;
                    let m2w = vb.pp("out").get((ne, inter, h), "weight")?;
                    let m2b = vb.pp("out").get((ne, h), "bias")?;
                    (n, gw, gb, m1w, m1b, m2w, m2b)
                }
                Naming::Hf => {
                    // post_attention_layernorm sits at block-level, not inside mlp.
                    let n = block_vb.pp("post_attention_layernorm").get(h, "weight")?;
                    let gw = vb.pp("router").get((ne, h), "weight")?;
                    let gb = vb.pp("router").get(ne, "bias")?;
                    // HF MoE serialization stores experts in the same [E, in, out]
                    // layout as gpt-oss, so we can load directly without a transpose.
                    let m1w = vb.pp("experts").get((ne, h, 2 * inter), "gate_up_proj")?;
                    let m1b = vb.pp("experts").get((ne, 2 * inter), "gate_up_proj_bias")?;
                    let m2w = vb.pp("experts").get((ne, inter, h), "down_proj")?;
                    let m2b = vb.pp("experts").get((ne, h), "down_proj_bias")?;
                    (n, gw, gb, m1w, m1b, m2w, m2b)
                }
            };

        let norm = RmsNorm::new(norm_scale, config.rms_norm_eps);
        let gate = Linear::new(gate_w, Some(gate_b));

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
        let (token_idx_t, weights_t, bucket_offsets) =
            time_stage(Stage::ForwardMoeRoute, || -> Result<_> {
                let gate_logits = self.gate.forward(&normed)?; // [T, num_experts]
                // candle 0.10 has no on-device topk and narrow() requires
                // host-side usize, so per-expert dispatch must read the
                // gating logits to host. On Metal this synchronizes the
                // command queue 8× per forward (~15ms/sync) — see
                // BENCHMARKS.md. The forward already runs in f32 so no cast
                // is needed before to_vec2.
                let gate_logits_f32 = time_stage(Stage::ForwardMoeRouteSync, || -> Result<_> {
                    Ok(gate_logits.to_vec2::<f32>()?)
                })?;
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

                // Build a single dispatch tensor per layer rather than one per
                // expert. sort_by_key is stable, so within each expert's bucket
                // tokens stay in ascending index order.
                let total = t * k;
                let mut order: Vec<usize> = (0..total).collect();
                order.sort_by_key(|&i| indices_flat[i]);

                let mut sorted_token_idx: Vec<u32> = Vec::with_capacity(total);
                let mut sorted_weights: Vec<f32> = Vec::with_capacity(total);
                let mut bucket_offsets: Vec<usize> = vec![0usize; self.num_experts + 1];
                for &i in &order {
                    sorted_token_idx.push((i / k) as u32);
                    sorted_weights.push(weights_flat[i]);
                    bucket_offsets[indices_flat[i] as usize + 1] += 1;
                }
                for e in 0..self.num_experts {
                    bucket_offsets[e + 1] += bucket_offsets[e];
                }

                // Two host->device transfers for dispatch metadata per layer
                // instead of ~2 * num_active_experts. narrow() below is
                // metadata-only.
                let token_idx_t = Tensor::from_vec(sorted_token_idx, total, device)?;
                let weights_t =
                    Tensor::from_vec(sorted_weights, (total, 1), device)?.to_dtype(dtype)?;
                Ok((token_idx_t, weights_t, bucket_offsets))
            })?;

        let out = time_stage(Stage::ForwardMoeExpert, || -> Result<_> {
            let mut out = Tensor::zeros((t, h), dtype, device)?;
            for e in 0..self.num_experts {
                let start = bucket_offsets[e];
                let n_e = bucket_offsets[e + 1] - start;
                if n_e == 0 {
                    continue;
                }
                let token_idx = token_idx_t.narrow(0, start, n_e)?;
                let weights_col = weights_t.narrow(0, start, n_e)?;

                let x_e = normed.index_select(&token_idx, 0)?; // [n_e, hidden]

                // Indexing along the leading dim of a contiguous
                // [num_experts, ..., ...] tensor yields a contiguous slice;
                // candle's matmul accepts the slice directly, so the prior
                // .contiguous() copy was redundant.
                let w1 = self.mlp1_weight.i(e)?; // [hidden, 2*intermediate]
                let b1 = self.mlp1_bias.i(e)?; // [2*intermediate]
                let h1 = x_e.matmul(&w1)?.broadcast_add(&b1)?;
                let h1 = swiglu(&h1, self.swiglu_limit)?; // [n_e, intermediate]

                let w2 = self.mlp2_weight.i(e)?; // [intermediate, hidden]
                let b2 = self.mlp2_bias.i(e)?; // [hidden]
                let h2 = h1.matmul(&w2)?.broadcast_add(&b2)?;

                let weighted = h2.broadcast_mul(&weights_col)?; // [n_e, hidden]
                out = out.index_add(&token_idx, &weighted, 0)?;
            }
            Ok(out)
        })?;

        x.add(&out).map_err(Into::into)
    }
}

fn top_k(values: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    let mut paired: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    if k < paired.len() {
        paired.select_nth_unstable_by(k, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
        });
        paired.truncate(k);
    }
    paired.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
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

/// Bidirectional sliding-window mask: query i attends to keys in [i-window, i+window].
/// The HF config's `sliding_window` value is the *half-width*; total window size is
/// 2*window + 1. The opf runtime requires `bidirectional_context=true` for all checkpoints.
fn bidirectional_sliding_mask(
    t: usize,
    window: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let neg_inf = f32::NEG_INFINITY;
    let mut data = vec![0f32; t * t];
    for i in 0..t {
        let lo = i.saturating_sub(window);
        let hi = (i + window).min(t - 1);
        for j in 0..t {
            if j < lo || j > hi {
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
    // Cap the precomputed table at a generous but bounded length. The model
    // architecture supports up to 131072 positions, but the table is
    // (max_pos, head_dim/2) f32 -> at 16384 the table is 16384*32*2*4 = ~4 MiB
    // per head_dim/2 slice (cos+sin), trivial. We rarely see >2k tokens in
    // production but raise the ceiling for safety on cloud-log inputs.
    let max_pos = ((initial_ctx * scaling_factor) as usize).min(16384);

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
