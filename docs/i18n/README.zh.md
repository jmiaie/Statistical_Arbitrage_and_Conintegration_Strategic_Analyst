<div align="center">

# quant-pairs-lab

### 动态统计套利 · 协整 · 风险感知回测

*基于卡尔曼滤波的配对交易研究框架，包含 Fama–French 因子归因与平方根市场冲击模型。*

[English](../../README.md) · [Español](./README.es.md) · [中文](./README.zh.md) · [日本語](./README.ja.md) · [Français](./README.fr.md)

</div>

---

## 1. 概述

`quant-pairs-lab` 是一个机构级统计套利研究平台，基于三项区别于教科书示例的核心原则：

- **自适应对冲比率。** 使用卡尔曼滤波替代静态 OLS 回归，使配对资产之间的关系能够随市场动态演化。
- **纯特质性 Alpha。** 通过 Fama–French 三因子模型分解收益，验证市场中性并将真实技能与风格暴露分离。
- **诚实的经济学。** 引入非线性（平方根）交易成本模型与延迟敏感性分析，在确认任何 P&L 之前先压力测试策略容量。

## 2. 量化方法

**A. 配对筛选与协整。** 行业中性、流动性过滤的股票池；Engle–Granger 两步法用于配对，Johansen 检验用于多元篮子；ADF / KPSS 验证 I(0) 平稳性。

**B. 卡尔曼滤波。** 状态空间形式让对冲比率 $\beta_t$ 在不引入前瞻偏差的前提下平滑穿越市场状态切换：

$$y_t = \beta_t x_t + \alpha_t + \varepsilon_t, \qquad \beta_t = \beta_{t-1} + \eta_t$$

**C. 风险与因子归因。** 策略收益对 Fama–French 因子回归；截距 $\alpha$ 为核心指标，各因子 Beta 作为诊断护栏，确保策略不是隐性的市场、规模或价值押注。

## 3. 执行与交易成本分析

| 组件 | 模型 | 目的 |
|---|---|---|
| **滑点** | 平方根定律 | 规模化下的真实冲击 |
| **延迟** | P&L 衰减 vs. 执行延迟（毫秒） | Alpha 半衰期 |
| **容量** | Sharpe 退化曲线 vs. 名义本金 | 最大可部署 AUM |
| **融券成本** | 空头返点折扣 | 净融资成本后收益 |

## 4. 关键指标

净 Sharpe · 最大回撤 · 信息系数 (IC) · 滚动因子 Beta · 换手率与胜率。

## 5. 快速开始

```bash
git clone https://github.com/jmiaie/quant-pairs-lab.git
cd quant-pairs-lab
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook research/main_backtest.ipynb
```

## 6. 许可证

版权所有 © 2026 Jeff Milam, MBA。保留所有权利。代码为专有财产，未经授权的复制或分发严格禁止。
