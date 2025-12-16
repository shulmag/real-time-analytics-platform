
```python
Project Aeacus
Created by Gil on 9/04/2025
Updated by Gil on 10/01/2025
```
# Bond2Vec: A Multi‑Task Learning Framework for Municipal Bond Embeddings
**Author:** Gil Shulman, ficc.ai  
**Contact:** gil@ficc.ai

## Summary

Bond2Vec learns **behavior-aware embeddings** for municipal bonds from **temporally aligned trade histories** and **reference data**, using a **Siamese neural network** trained with a **cosine-contrastive loss**. The resulting 128-d vectors power:

* **New-issue pricing** (nearest-neighbor, locally weighted regression)
* **Liquidity score** (neighbor density × recency × volume stability)
* **Pricing for bonds that haven’t traded in a long time**
* **Arbitrage scanning** (similarity high, spread gap high)
* **Portfolio analysis & clustering** (explainable behavioral clusters)
* **“Which bond should I sell?”** (liquidity × diversification × mispricing)

No “cold-start” phrasing: we focus on **illiquid** names and bonds that **haven’t traded in a long time**.

## Table of Contents

1. [Why Bond2Vec?](#why-bond2vec)
2. [How it works](#how-it-works)
3. [Data schema](#data-schema)
4. [Training](#training)
5. [Inference](#inference)
6. [Applications](#applications)
7. [Math (plain-English)](#math-plain-english)
8. [Repository structure](#repository-structure)
9. [Roadmap](#roadmap)
10. [References](#references)

## Why Bond2Vec?

Municipal bonds **often don’t trade for long periods** and differ widely by structure and size. Pure catalog similarity (coupon, rating, etc.) can miss **how bonds actually trade together**. Bond2Vec learns a space where **cosine similarity ≈ co-movement** in trading behavior.

**Key differences vs Product2Vec:**

* Labels come from **short trade-history cosine** within **tight time windows**, not browsing/purchase logs.
* **Temporal alignment** ensures pairs reflect the same market regime.
* Encoder outputs **L2-normalized 128-d embeddings**; similarity = **dot product**.

## How it works

### 1) Trade-history labeling

* Each CUSIP: **5×6 matrix** of recent prints
  `[yield_spread, treasury_spread, log_par_traded, trade_type_1, trade_type_2, log_seconds_ago]`
* Flatten to 30-d vector; compute cosine `s_hist` between CUSIPs in the **same 30s window**.
* Label **positive** if `s_hist > 0.5`, **negative** if `s_hist < 0.2`.

### 2) Siamese encoder

* Input: engineered reference features (~hundreds).
* MLP: `512 → 256 → 256 → 128` + **L2 normalize**.
* Pair similarity = dot product (cosine).
* **Loss:** contrastive on cosine distance `D = 1 − cos` with margin `m ≈ 0.5`.

### 3) Multi-task ready

Currently: cosine head.
Future: add unary/binary heads (e.g., “trades in next Δt?”, rating drift risk).

## Data schema

### Reference features

* Coupon, coupon type, maturity, call/refund/sink schedule
* State & tax flags, purpose class/sub-class, security type
* Outstanding/issuance amounts, accrual dates
* Ratings: string **and** numeric score (AAA→22 … NR→0)
* Quantity (log-scaled), settlement delta, A/E ratio

### Trade-history

* 5×6 matrix (see above) → flattened to 30-d, cosine for label mining.

## Training

```python
from siamese_network import run_training_pipeline

base_net, artifacts, hist = run_training_pipeline(
    features_df=trades_with_ref_features_df,
    pairs_df=temporally_labeled_pairs_df,
    embedding_dim=128,
    epochs=100,
)
```

**Pipeline**

1. Build temporally aligned (A,B,label) pairs.
2. Engineer features & fit artifacts (RobustScaler + LabelEncoders).
3. Train Siamese with cosine-contrastive loss; early stop.
4. Saves:

   * `cusip_embedding_model_temporal.keras`
   * `feature_artifacts_temporal.pkl`



## Inference

```python
from redis_embedder import RedisCusipEmbedder

embedder = RedisCusipEmbedder(
    model_path="cusip_embedding_model_temporal.keras",
    artifacts_path="feature_artifacts_temporal.pkl",
    redis_host="localhost"
)
vec, meta = embedder.embed("CUSIP123", trade_type="P", quantity=100)
```

**Returns:**

* 128-d embedding (unit norm)
* Metadata snapshot

Use for pricing, liquidity, arbitrage, clustering, or sell-decision tooling.



## Applications

* **New-issue pricing:** k-NN in embedding space → locally weighted regression on spreads vs curve.
* **Liquidity score:** neighbor density − recency penalty + volume z-score.
* **Price illiquids:** regression on neighbors + prediction intervals.
* **Arbitrage scanner:** similar bonds with abnormal spread gaps.
* **Portfolio clustering:** spherical k-means, centroid distance = diversification.
* **Which bond to sell:** convex score of liquidity, diversification, mispricing.



## Math (plain-English)

* **Cosine head:** embeddings are L2-normalized; dot product = cosine ∈ [-1,1].
  *Meaning:* aligned vectors → 1; orthogonal → 0; opposite → −1.
* **Contrastive loss:**

  * Positives: `(1 − cos)^2` → pull closer.
  * Negatives: penalize if `D < m`; push until at least margin `m`, then stop.
* **Temporal alignment:** labels built only within same 30s window → avoids regime mix.



## Repository structure

```
.
├─ feature_engineering.py        # Scaling + encoders; date/coupon/rating logic
├─ bond_embedding_helpers.py     # 30/360, accrual, rating map, trade-history cosine
├─ pair_generation.py            # 30s windows → (A,B,label)
├─ siamese_network.py            # Base MLP, L2 normalize, cosine head, loss
├─ redis_embedder.py             # Point-in-time ref data → embedding
├─ intro_to_bond2vec.ipynb       # Notebook walkthrough
└─ README.md                     # This file
```



## Roadmap

* Tune thresholds (positive/negative) and window length.
* Add unary/binary heads (trade probability, rating drift, co-trade).
* Export approximate-NN index for real-time retrieval.
* Confidence metrics for pricing recs.



## References

* Mikolov et al., *Distributed Representations of Words and Phrases and their Compositionality*, NIPS 2013.
* Bromley et al., *Signature Verification using a “Siamese” Time Delay Neural Network*, NIPS 1993.
* Hadsell et al., *Dimensionality Reduction by Learning an Invariant Mapping*, 2006.
* Huang et al., *Deep Structured Semantic Models*, CIKM 2013.
* Caruana, *Multitask Learning*, 1997.
* London et al., *Product2Vec: A Multi-Task Learning Framework for Cold-start Product Recommendation*.

