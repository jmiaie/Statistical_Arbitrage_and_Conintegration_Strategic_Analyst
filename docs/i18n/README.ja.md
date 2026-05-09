<div align="center">

# quant-pairs-lab

### 動的統計的裁定取引 · 共和分 · リスク認識バックテスト

*カルマンフィルタによるペアトレード研究基盤。Fama–French ファクター帰属と平方根マーケットインパクトモデルを内蔵。*

[English](../../README.md) · [Español](./README.es.md) · [中文](./README.zh.md) · [日本語](./README.ja.md) · [Français](./README.fr.md)

</div>

---

## 1. エグゼクティブサマリー

`quant-pairs-lab` は、機関投資家水準の統計的裁定取引研究スタックです。教科書的なペアトレードと一線を画す三つの原則に基づいて設計されています。

- **適応的ヘッジ比率。** 静的 OLS をカルマンフィルタに置き換え、ペア資産間の関係が市場とともに進化するようにします。
- **純粋な固有アルファ。** Fama–French 3 ファクターモデルでリターンを分解し、マーケットニュートラリティを確認するとともに、スタイル傾斜からスキルを切り離します。
- **正直な経済性。** 非線形（平方根）取引コストモデルとレイテンシ感度分析により、P&L を喜ぶ前にキャパシティをストレステストします。

## 2. 定量手法

**A. ペア選定と共和分。** セクター中立かつ流動性でフィルタリングされたユニバース、Engle–Granger 法と Johansen 検定、ADF / KPSS による I(0) 定常性確認。

**B. カルマンフィルタ。** 状態空間表現により、ヘッジ比率 $\beta_t$ は先読みバイアスなしにレジーム変化を滑らかに通過します。

$$y_t = \beta_t x_t + \alpha_t + \varepsilon_t, \qquad \beta_t = \beta_{t-1} + \eta_t$$

**C. リスクとファクター帰属。** 戦略リターンを Fama–French ファクターに回帰。切片 $\alpha$ が主要指標、各ベータは戦略がマーケット・サイズ・バリューの隠れた賭けではないことを保証する診断ガードレールです。

## 3. 執行と取引コスト分析 (TCA)

| 要素 | モデル | 目的 |
|---|---|---|
| **スリッページ** | 平方根則 | 規模における現実的なインパクト |
| **レイテンシ** | 執行遅延に対する P&L 減衰 | アルファの半減期 |
| **キャパシティ** | 想定元本に対する Sharpe 劣化曲線 | 最大運用可能 AUM |
| **貸株コスト** | ショートリベートの控除 | 資金調達後の純リターン |

## 4. 主要 KPI

ネット Sharpe · 最大ドローダウン · インフォメーションコア (IC) · ローリングファクターベータ · 回転率と勝率。

## 5. クイックスタート

```bash
git clone https://github.com/jmiaie/quant-pairs-lab.git
cd quant-pairs-lab
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook research/main_backtest.ipynb
```

## 6. ライセンス

Copyright © 2026 Jeff Milam, MBA. All rights reserved. 本コードはプロプライエタリであり、無許可の複製・配布は固く禁じられています。
