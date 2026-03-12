# Q2 - Agent 驱动的金融建模系统（方向 A）

本目录提供一个可复现的 **Agent 化特征工程系统**，自动完成：

1. 特征诊断（缺失/异常/分布）
2. 特征清理（自动决策并执行）
3. 特征评估（互信息 + 树模型重要性 + 冗余检测）
4. Top50 特征筛选（多轮）
5. 可视化与报告输出

## 目录结构

- `Q2.ipynb`：主 Notebook（已执行版）
- `Q2.md`：系统设计文档
- `code/agent_feature_system.py`：Agent 系统代码
- `output/`：运行结果（日志、报告、图表）

## 运行方式

```bash
python -m pip install -r requirements.txt
jupyter nbconvert --to notebook --execute Q2/Q2.ipynb --output Q2.ipynb
```

## 关键输出索引（Notebook 内）

- 数据概况输出
- 训练过程输出（特征筛选与建模）
- 评估指标输出（AUC/Precision/Recall/F1）
- 可视化输出（混淆矩阵/ROC）
- 数据泄漏检查输出
