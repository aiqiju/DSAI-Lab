# DSE-0001 — Seed Development Intrinsic Dimension (ID) Study (Pre-registration)

**DS 种子**：胚胎/种子发育的内在维度 D(t) 随时间上升，并在器官形成期出现拐点。
**时间戳**：2025-11-02 12:37 JST


**Hypothesis**: During embryonic/seed development, intrinsic dimension D(t) increases monotonically with at least one inflection (organogenesis).

## Variables
- Input: time-ordered samples (single-cell expression or morphology embeddings)
- Output: ID estimates per timepoint: {PCA-based rank, TwoNN-ID, Morisita dimension}

## Metrics & Tests
- Monotonicity: Kendall's τ trend test (α=0.05)
- Inflection: piecewise linear fit or segmented regression; BIC/ΔAIC for model choice

## Falsification
- If multiple datasets show non-monotonic D(t) and no robust inflection under sensitivity analysis → reject current DEE form, revise equations or include constraints.

## Minimal Procedure (≤30 min)
1) Put a small CSV with rows=samples, cols=features, and a 'time' column into `data/`.
2) Run `python code/compute_id_curve.py --csv data/your.csv --timecol time --out results/`.
3) Inspect `results/id_curve.csv` and the generated plot; record conclusion below.

## Conclusion (to be filled after run)
- Support / Not support / Inconclusive (reasons)

## Next Step
- If inconclusive: increase samples, try another ID estimator, control confounders.
