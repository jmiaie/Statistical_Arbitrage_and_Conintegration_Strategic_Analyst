<div align="center">

# quant-pairs-lab

### Arbitrage Statistique Dynamique · Cointégration · Backtesting Conscient du Risque

*Une infrastructure de recherche de *pairs trading* avec filtre de Kalman, attribution Fama–French et modèle d'impact de marché en racine carrée.*

[English](../../README.md) · [Español](./README.es.md) · [中文](./README.zh.md) · [日本語](./README.ja.md) · [Français](./README.fr.md)

</div>

---

## 1. Résumé Exécutif

`quant-pairs-lab` est une plateforme d'arbitrage statistique de qualité institutionnelle, construite sur trois principes qui distinguent la pratique professionnelle des exemples académiques :

- **Ratios de couverture adaptatifs.** Un filtre de Kalman remplace la régression OLS statique afin que la relation entre les actifs appariés évolue avec le marché.
- **Alpha idiosyncratique pur.** Les rendements sont décomposés selon le modèle Fama–French à 3 facteurs pour confirmer la neutralité au marché et isoler la véritable compétence des biais de style.
- **Économie honnête.** Un modèle de coûts de transaction non linéaire (racine carrée) et une analyse de sensibilité à la latence stressent la capacité avant toute célébration du P&L.

## 2. Méthodologie Quantitative

**A. Sélection de paires et cointégration.** Univers neutre par secteur, validation statistique via Engle–Granger et test de Johansen, confirmation de la stationnarité I(0) du spread.

**B. Filtre de Kalman.** Une formulation espace-d'état permet au ratio de couverture $\beta_t$ de dériver à travers les changements de régime sans biais d'anticipation :

$$y_t = \beta_t x_t + \alpha_t + \varepsilon_t, \qquad \beta_t = \beta_{t-1} + \eta_t$$

**C. Attribution du risque.** Les rendements de la stratégie sont régressés sur les facteurs Fama–French ; l'intercept $\alpha$ est l'indicateur principal et les bêtas servent de garde-fous diagnostiques.

## 3. Exécution et Analyse des Coûts de Transaction

| Composant | Modèle | Objectif |
|---|---|---|
| **Slippage** | Loi en racine carrée | Impact réaliste à grande échelle |
| **Latence** | Décroissance du P&L vs. délai (ms) | Demi-vie de l'alpha |
| **Capacité** | Courbe Sharpe vs. notionnel | AUM maximal déployable |
| **Coût d'emprunt** | Ajustement du *short rebate* | Rendement net de financement |

## 4. Indicateurs Clés

Sharpe net · Drawdown maximal · Coefficient d'Information (IC) · Bêtas factoriels glissants · Rotation et taux de succès.

## 5. Démarrage Rapide

```bash
git clone https://github.com/jmiaie/quant-pairs-lab.git
cd quant-pairs-lab
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook research/main_backtest.ipynb
```

## 6. Licence

Copyright © 2026 Jeff Milam, MBA. Tous droits réservés. Code propriétaire ; toute copie ou distribution non autorisée est strictement interdite.
