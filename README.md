# Salt Cavern ML Template

面向“盐穴长期变形与体积收缩率”研究的 Python 项目模板，包含：

- 数据结构约定（case-level + case-year）
- 训练脚本（预测模型）
- 分型脚本（长期演化模式识别）
- 解释分析脚本（主控因素与参数耦合）
- 论文级绘图规范与统一风格

## 1. 项目结构

```text
.
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── clusters/
│   ├── embeddings/
│   ├── explain/
│   ├── figures/
│   ├── models/
│   └── tables/
├── scripts/
│   ├── 01_prepare_data.py
│   ├── 02_pattern_clustering.py
│   ├── 03_train_predictor.py
│   ├── 04_explain_coupling.py
│   └── 05_plot_paper_figures.py
└── src/salt_ml/
    ├── data_utils.py
    ├── feature_engineering.py
    ├── modeling.py
    ├── plotting.py
    └── typing_defs.py
```

## 2. 数据输入格式

### `data/raw/case_static.csv`
每行一个 case，建议包含：

- `case_id`
- 几何参数：`DT, DB, DHR, TR, SH, JH, CD, ...`
- 力学参数：`Ey, vy, Cy, Fy, Ty, Ay, ny, En, vn, ...`
- 运行参数：`LP, HP, CF, CV, ...`

### `data/raw/case_year.csv`
每行是 case-year：

- 主键：`case_id, year`
- 时序响应：`VolShrinkRatio, TotalVolumn, MaxDisp, MaxAbsUx, MaxAbsUz, MinPrincipal, ...`

## 3. 快速开始

```bash
python -m pip install -e .
python scripts/01_prepare_data.py --config configs/default.yaml
python scripts/02_pattern_clustering.py --config configs/default.yaml
python scripts/03_train_predictor.py --config configs/default.yaml
python scripts/04_explain_coupling.py --config configs/default.yaml
python scripts/05_plot_paper_figures.py --config configs/default.yaml
```

## 4. 模板思想

1. **分型后预测**：先做长期模式识别，再做条件预测。
2. **可解释性优先**：SHAP + 交互分析 + 统计检验。
3. **统一绘图规范**：颜色、字体、线宽、导出分辨率统一。

## 5. 可扩展建议

- 加入 TFT/LSTM 时序模型作为 `modeling.py` 的可选后端。
- 使用 Soft-DTW + HDBSCAN 替换基础聚类。
- 引入 Conformal Prediction 输出置信区间。
