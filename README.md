# AIoT Physics RF-3DGS

面向窄带无源反散射定位的物理RF-3DGS实现。系统不训练 `Signal -> Position` 回归器，而是通过静态环境Gaussian、动态物体Gaussian及 `CW -> Tag -> RX` 双基地物理渲染反演物体中心和置信度。

## 数据与环境

服务器需要RTX 4090、可用的PyTorch CUDA环境和WRF-GS `diff_gaussian_rasterization` 扩展。原始数据放在Git仓库之外，目录结构见 `data/README.md`。

```bash
conda activate wrfgsplus
pip install -r requirements-rf3dgs.txt
python -m pytest
```

## 运行

先执行冒烟测试，确认数据、CUDA和完整反演链路正常：

```bash
DATA_ROOT=/data/aiot/raw bash scripts/run_smoke.sh
```

正式训练使用20万静态Gaussian、0.25 m物理特征网格及五阶段优化：

```bash
DATA_ROOT=/data/aiot/raw RUN_ROOT=/runs/rf3dgs/run_001 bash scripts/run_full.sh
```

## 输出与复现

`RUN_ROOT` 保存模型、训练清单和评估结果。`evaluation/metrics.json` 报告四组消融、平均/中位/P90误差、`SR@1m / SR@3m / SR@5m` 和R90覆盖率；训练清单记录输入SHA-256、Git commit、CUDA/PyTorch版本、随机种子及完整配置。

