# CD2R: Counterfactual Dual-Channel Causal Disentanglement for Debiased Doctor Recommendation

Due to patient–doctor data privacy constraints, the released datasets have been anonymized/encrypted.

---

## 1. Model Architecture

| Module | Role |
|--------|------|
| Medical-channel GNN | Interaction graph + KG fused graph → specialty match score `score_med` |
| Popularity-channel GNN | Interaction graph → exposure/popularity bias score `score_conf` |
| Reasoning-chain alignment \(L_{align}\) | Align RAG chain embeddings with the medical channel (training-time; weights are shipped) |
| Training score | \(\hat y_{train}=score_{med}+score_{conf}\) |
| Causal debiasing (refined) | Remove harmful conf-over-med effects, with medical confidence \(M\) protection |



**Ablations:**

| Variant | Removed component |
|---------|-------------------|
| Full | — |
| Abl-01 | No LLM semantic supervision |
| Abl-02 | Dual channel (medical channel only) |
| Abl-03 | Causal debiasing (\(\alpha=0\), reuse Full weights) |
| Abl-04 | Remove KG enhancement edges |

---

## 2. Directory Layout

```
CD2R_release/
├── README.md / README_zh.md
├── requirements.txt
├── config.py                 # global hyperparameters and paths
├── cd2r/                     # model and evaluation
│   ├── model/                # dual/single-channel GNN, causal debiasing
│   ├── data/                 # data loading
│   └── eval/                 # counterfactual inference, metrics, engine
├── preprocess/               # data processing scripts
├── scripts/
│   ├── eval_main.py          # main experiments on four datasets
│   ├── eval_ablation.py      # mixed ablations
│   └── run_all_eval.sh
├── artifacts/{dataset}/{variant}/   # model.pth
└── results/                  # evaluation outputs (metrics.json)
```



---

## 3. Main Hyperparameters

| Name | Default | Notes |
|------|---------|-------|
| `SEED` | 42 | T2 downsampling |
| `EMB_DIM` | 64 | embedding dimension |
| `N_LAYERS` | 2 | GNN layers |
| `EVAL_K` | [5, 10] | |
| `CAUSAL_PROFILE` | `refined` | `m_sharpness=3.0`, `te_margin=0.0` |
| `CAUSAL_ALPHA` | **0.5** | \(\alpha_0\) |
| `LAMBDA_ALIGN` | 0.15 | Full training; Abl-01 trained with `0` |

---

## 4. How to Run

```bash
git clone https://github.com/Yeechen-ai/CD2R2.0.git
cd CD2R2.0
pip install -r requirements.txt

# main experiments
python scripts/eval_main.py --dataset all

# ablations
python scripts/eval_ablation.py --ablation all

# one-shot
bash scripts/run_all_eval.sh
```

Single-dataset / single-ablation examples:

```bash
python scripts/eval_main.py --dataset mixed
python scripts/eval_main.py --dataset resp
python scripts/eval_ablation.py --ablation abl01
```


---

## 5. Outputs

| Experiment | Path |
|------------|------|
| Main results | `results/{mixed,resp,oral,inf}/full/metrics.json` |
| Ablations | `results/mixed/abl0{1,2,3,4}/metrics.json` |
