<div align="center">

# quant-pairs-lab

### Arbitraje Estadístico Dinámico · Cointegración · Backtesting Consciente del Riesgo

*Una infraestructura de investigación para *pairs trading* con filtro de Kalman, atribución Fama–French y modelo de impacto de mercado de raíz cuadrada.*

[English](../../README.md) · [Español](./README.es.md) · [中文](./README.zh.md) · [日本語](./README.ja.md) · [Français](./README.fr.md)

</div>

---

## 1. Resumen Ejecutivo

`quant-pairs-lab` es una plataforma de arbitraje estadístico de nivel institucional, construida sobre tres principios que distinguen a la práctica profesional de los ejemplos de manual:

- **Ratios de cobertura adaptativos.** Un filtro de Kalman reemplaza la regresión OLS estática para que la relación entre activos pareados evolucione con el mercado.
- **Alfa idiosincrático puro.** Los retornos se descomponen contra el modelo Fama–French de 3 factores para confirmar la neutralidad al mercado y aislar la habilidad de los sesgos de estilo.
- **Economía honesta.** Un modelo de costos de transacción no lineal (raíz cuadrada) y un análisis de sensibilidad a la latencia ponen a prueba la capacidad antes de celebrar cualquier P&L.

## 2. Metodología Cuantitativa

**A. Selección de pares y cointegración.** Universo neutral por sector, validación estadística mediante Engle–Granger y la prueba de Johansen, confirmación de estacionariedad I(0) en el spread.

**B. Filtro de Kalman.** Formulación en espacio de estados que permite que el ratio de cobertura $\beta_t$ derive suavemente a través de cambios de régimen sin sesgo de anticipación:

$$y_t = \beta_t x_t + \alpha_t + \varepsilon_t, \qquad \beta_t = \beta_{t-1} + \eta_t$$

**C. Atribución de riesgo.** Los retornos de la estrategia se regresan contra los factores Fama–French; el intercepto $\alpha$ es la cifra principal y los betas funcionan como guardarraíles diagnósticos.

## 3. Ejecución y Análisis de Costos de Transacción

| Componente | Modelo | Propósito |
|---|---|---|
| **Slippage** | Ley de raíz cuadrada | Impacto realista a escala |
| **Latencia** | Decaimiento de P&L vs. retraso (ms) | Vida media del alfa |
| **Capacidad** | Curva Sharpe vs. nocional | AUM máximo desplegable |
| **Costo de préstamo** | Ajuste por rebate de short | Retornos netos de financiamiento |

## 4. Indicadores Clave

Sharpe neto · Drawdown máximo · Coeficiente de Información (IC) · Betas factoriales rodantes · Turnover y tasa de aciertos.

## 5. Inicio Rápido

```bash
git clone https://github.com/jmiaie/quant-pairs-lab.git
cd quant-pairs-lab
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook research/main_backtest.ipynb
```

## 6. Licencia

Copyright © 2026 Jeff Milam, MBA. Todos los derechos reservados. Código propietario; la copia o distribución no autorizada está estrictamente prohibida.
