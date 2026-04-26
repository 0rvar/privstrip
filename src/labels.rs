use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    Background,
    B,
    I,
    E,
    S,
}

#[derive(Debug, Clone)]
pub struct LabelInfo {
    pub id2label: Vec<String>,
    pub boundary: Vec<Boundary>,
    pub span_label: Vec<Option<String>>,
    pub background_idx: usize,
}

impl LabelInfo {
    pub fn from_config(config_path: &Path) -> Result<Self> {
        #[derive(Deserialize)]
        struct Cfg {
            id2label: BTreeMap<String, String>,
        }
        let raw = std::fs::read_to_string(config_path)
            .with_context(|| format!("read {}", config_path.display()))?;
        let cfg: Cfg = serde_json::from_str(&raw)?;

        let mut pairs: Vec<(usize, String)> = cfg
            .id2label
            .into_iter()
            .map(|(k, v)| Ok((k.parse::<usize>()?, v)))
            .collect::<Result<Vec<_>>>()?;
        pairs.sort_by_key(|(i, _)| *i);

        let n = pairs.len();
        let mut id2label = vec![String::new(); n];
        let mut boundary = vec![Boundary::Background; n];
        let mut span_label = vec![None; n];
        let mut background_idx = 0usize;

        for (idx, label) in pairs {
            id2label[idx] = label.clone();
            if label == "O" {
                boundary[idx] = Boundary::Background;
                background_idx = idx;
            } else if let Some((tag, rest)) = label.split_once('-') {
                boundary[idx] = match tag {
                    "B" => Boundary::B,
                    "I" => Boundary::I,
                    "E" => Boundary::E,
                    "S" => Boundary::S,
                    _ => anyhow::bail!("unknown boundary tag in label {label}"),
                };
                span_label[idx] = Some(rest.to_string());
            } else {
                anyhow::bail!("malformed label {label}");
            }
        }
        Ok(Self {
            id2label,
            boundary,
            span_label,
            background_idx,
        })
    }

    pub fn num_classes(&self) -> usize {
        self.id2label.len()
    }
}
