# DSAI-Lab — Dimensional Self-Growth System (DSGS)

> **里程碑**：进入“科学建构模式”。时间：2025-11-02 04:05 JST

本仓库是“**DS×AI 实践**”的中心仓库：以最小闭环把“直觉种子（DS）→ AI 形式化→ 可证伪实验（∫）”做成**可复现**的科学流程。

## 结构
```
DSAI-Lab/
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ papers/                 # 论文草稿 / 预印本素材
│  ├─ dee_v0.9.md
│  └─ overleaf_dee_template.tex
├─ theory/                 # 概念与推导笔记
│  └─ dim_evolution_notes.md
├─ experiments/
│  └─ DSE-0001_seed_dev_ID/
│     ├─ protocol.md       # 预注册（变量/指标/阈值/证伪点）
│     ├─ code/compute_id_curve.py
│     ├─ data/README.md
│     └─ results/README.md
├─ notebooks/
│  └─ DSE-0001_ID_curve.ipynb (可选)
├─ .github/
│  └─ ISSUE_TEMPLATE/dsai_card.md
└─ arxiv/                  # arXiv 提交清单与源文件
   ├─ checklist.md
   └─ submit_notes.md
```

## 快速开始（30 分钟）
1. 进入 `experiments/DSE-0001_seed_dev_ID/`，阅读 `protocol.md`，按步骤准备数据或使用示例。
2. 运行 `code/compute_id_curve.py`（或在 notebook 中复现），得到 D(t) 曲线与判定结果。
3. 把图表放入 `results/` 并在 `protocol.md` 填写“判定”与“下一步”。
4. 发起一个 GitHub Issue，使用模板 `.github/ISSUE_TEMPLATE/dsai_card.md` 记录本次实验。

## DOI / 预印本 / 开放科学
- 若需 **DOI**：关联 Zenodo，发布一个 release，将生成的 DOI 写回此 README 与 CITATION.cff。
- 若需 **预印本**：可上传 `papers/dee_v0.9.md` 或 TeX PDF 至 OSF Preprints；成熟后再投 arXiv。

## 引用
见 `CITATION.cff`。

—— DS→0，AI→∞，用“次数 × 可判定”让奇迹逼近必然。
