# CD2R: Causal Dual-channel Debiased Recommendation

Final model release for reviewer evaluation (**eval-only**).

Chinese documentation: [`README_zh.md`](README_zh.md).

Covers Path-A main performance on **mixed / resp. / oral / Inf.** (T1/T2), and ablations on **mixed** only (Abl-01…04).

Metrics: **Recall / MRR / SMR @5/@10**.

---

## 1. Model (final)

| Module | Role |
|--------|------|
| Medical GNN | Interaction + KG fused graph → specialty match |
| Popularity GNN | Interaction graph → exposure bias |
| Chain align \(L_{align}\) | RAG chain embedding ↔ medical channel (training; weights shipped) |
| Train score | \(\hat y_{train}=score_{med}+score_{conf}\) |
| Causal debias (refined) | Remove harmful conf-over-med effect with confidence \(M\); illness boost \(\gamma\) |

**Ablations (mixed only):**

| Variant | Removed | Read column |
|---------|---------|-------------|
| Full | — | causal |
| Abl-01 | align loss \(\lambda_{align}=0\) | causal |
| Abl-02 | dual channel (med only) | train |
| Abl-03 | causal debias (\(\alpha=0\), Full weights) | train |
| Abl-04 | KG enhancement edges | causal |

---

## 2. Layout

```
CD2R_release/
├── README.md
├── README_zh.md
├── requirements.txt
├── config.py
├── cd2r/                 # model + eval
├── preprocess/           # Stage1–3 logic (needs private data; not runnable here)
├── scripts/
│   ├── eval_main.py
│   ├── eval_ablation.py  # mixed only
│   └── run_all_eval.sh
├── artifacts/
└── results/
```

Datasets in code: `mixed` / `resp` / `oral` / `inf` (docs: mixed / resp. / oral / Inf.).

Privacy: hashed IDs, opaque entity tokens; no raw CSV / TTL / chain text.

---

## 3. Parameters (explicit)

| Name | Default | Notes |
|------|---------|-------|
| `SEED` | 42 | T2 downsampling |
| `EMB_DIM` | 64 | |
| `N_LAYERS` | 2 | |
| `EVAL_K` | [5, 10] | |
| `CAUSAL_PROFILE` | `refined` | `m_sharpness=3.0`, `te_margin=0.0` |
| `CAUSAL_ALPHA` | **0.5** | \(\alpha_0\); Abl-03 uses `0.0` |
| `ADAPTIVE_ALPHA` | True | |
| `ADAPTIVE_ALPHA_BETA` | 2.0 | \(\beta\) |
| `ILLNESS_BOOST_GAMMA` | **4.0** | \(\gamma\); Abl-02 uses `0.0` |
| `PRIMARY_ILLNESS_BOOST` | **True** | boost uses primary illness only |
| `LAMBDA_ALIGN` | 0.15 | Full training; Abl-01 trained with `0` |

Overrides are baked per artifact in `artifacts/*/meta.json` (`causal_alpha`, `illness_boost_gamma`, `report_metrics`).

---

## 4. Run

```bash
git clone https://github.com/Yeechen-ai/CD2R2.0.git
cd CD2R2.0
pip install -r requirements.txt

python scripts/eval_main.py --dataset all
python scripts/eval_ablation.py --ablation all   # mixed only

bash scripts/run_all_eval.sh
```

```bash
python scripts/eval_main.py --dataset mixed
python scripts/eval_main.py --dataset resp
python scripts/eval_ablation.py --ablation abl01
```

Ablations for `resp` / `oral` / `inf` are **not** included and will error if requested.

---

## 5. Outputs

| Experiment | Path |
|------------|------|
| Main | `results/{mixed,resp,oral,inf}/full/metrics.json` |
| Ablation | `results/mixed/abl0{1,2,3,4}/metrics.json` |

---

## 6. Preprocess (not runnable for reviewers)

`preprocess/` keeps Stage1 split, Stage2 KG (+ specialty→symptom edges), and Stage3 RAG chain extraction logic.

Requires `CD2R_PRIVATE_DATA` (raw CSV + TTL + LLM). Without it, entry points raise. Use `artifacts/` for evaluation.

---

## 7. Expected metrics (primary)

### Main (causal)

| Dataset | Split | Recall@5 | Recall@10 | MRR@5 | MRR@10 | SMR@5 | SMR@10 |
|---------|-------|----------|-----------|-------|--------|-------|--------|
| mixed | T1 | 0.60000 | 0.83333 | 0.36778 | 0.39863 | 0.72896 | 0.63637 |
| mixed | T2 | 0.64516 | 0.82258 | 0.43038 | 0.45313 | 0.71576 | 0.62678 |
| resp. | T1 | 0.73864 | 0.92045 | 0.42803 | 0.45341 | 0.81400 | 0.78841 |
| resp. | T2 | 0.83333 | 0.83333 | 0.46574 | 0.46574 | 0.82266 | 0.79389 |
| oral | T1 | 0.76923 | 0.86538 | 0.41378 | 0.42620 | 0.85617 | 0.84011 |
| oral | T2 | 0.68182 | 0.86364 | 0.42197 | 0.44374 | 0.84654 | 0.81032 |
| Inf. | T1 | 0.84906 | 0.94340 | 0.60912 | 0.62190 | 0.84168 | 0.76261 |
| Inf. | T2 | 0.82143 | 0.92857 | 0.60417 | 0.61730 | 0.86461 | 0.79498 |

### mixed ablation (primary)

| Model | Split | Recall@5 | MRR@5 | SMR@5 | SMR@10 |
|-------|-------|----------|-------|-------|--------|
| Full | T1 | 0.60000 | 0.36778 | 0.72896 | 0.63637 |
| Abl-01 | T1 | 0.56667 | 0.35556 | 0.72896 | 0.63487 |
| Abl-02 | T1 | 0.41111 | 0.21870 | 0.58468 | 0.53236 |
| Abl-03 | T1 | 0.46667 | 0.24204 | 0.55763 | 0.52599 |
| Abl-04 | T1 | 0.60000 | 0.34222 | 0.55144 | 0.46624 |
| Full | T2 | 0.64516 | 0.43038 | 0.71576 | 0.62684 |
| Abl-01 | T2 | 0.59677 | 0.41935 | 0.71576 | 0.62455 |
| Abl-02 | T2 | 0.45161 | 0.24220 | 0.56292 | 0.52019 |
| Abl-03 | T2 | 0.50000 | 0.28360 | 0.52225 | 0.50853 |
| Abl-04 | T2 | 0.61290 | 0.34758 | 0.45266 | 0.37349 |
