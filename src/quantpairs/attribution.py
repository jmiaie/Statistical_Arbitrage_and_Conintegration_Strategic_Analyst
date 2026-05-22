"""Fama-French 3-factor attribution with Newey-West standard errors."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class FamaFrenchAttribution:
    """Result of regressing strategy returns on Fama-French factors."""

    alpha_annual: float
    alpha_tstat: float
    beta_mkt: float
    beta_smb: float
    beta_hml: float
    r_squared: float
    n_obs: int

    @classmethod
    def fit(
        cls,
        strategy_returns: pd.Series,
        ff_factors: pd.DataFrame,
        risk_free: pd.Series | None = None,
        annualisation: int = 252,
        nw_lags: int = 5,
    ) -> FamaFrenchAttribution:
        """Fit the FF3 model. `ff_factors` columns: Mkt-RF, SMB, HML."""
        required = ["Mkt-RF", "SMB", "HML"]
        missing = [c for c in required if c not in ff_factors.columns]
        if missing:
            raise ValueError(f"ff_factors missing columns: {missing}")

        df = pd.concat([strategy_returns.rename("ret"), ff_factors[required]], axis=1).dropna()
        if risk_free is not None:
            df = df.join(risk_free.rename("rf"), how="inner")
            y = df["ret"] - df["rf"]
        else:
            y = df["ret"]
        X = sm.add_constant(df[required])
        fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
        alpha_daily = float(fit.params.iloc[0])
        return cls(
            alpha_annual=alpha_daily * annualisation,
            alpha_tstat=float(fit.tvalues.iloc[0]),
            beta_mkt=float(fit.params["Mkt-RF"]),
            beta_smb=float(fit.params["SMB"]),
            beta_hml=float(fit.params["HML"]),
            r_squared=float(fit.rsquared),
            n_obs=int(fit.nobs),
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                ("Annualised Alpha", f"{self.alpha_annual:.2%}"),
                ("Alpha t-stat (NW)", f"{self.alpha_tstat:.2f}"),
                ("Market Beta", f"{self.beta_mkt:.3f}"),
                ("SMB Beta", f"{self.beta_smb:.3f}"),
                ("HML Beta", f"{self.beta_hml:.3f}"),
                ("R-squared", f"{self.r_squared:.3f}"),
                ("Observations", f"{self.n_obs}"),
            ],
            columns=["Metric", "Value"],
        )


def rolling_betas(
    strategy_returns: pd.Series, ff_factors: pd.DataFrame, window: int = 60
) -> pd.DataFrame:
    """60-day rolling FF3 betas for the diagnostics heatmap."""
    cols = ["Mkt-RF", "SMB", "HML"]
    df = pd.concat([strategy_returns.rename("ret"), ff_factors[cols]], axis=1).dropna()
    out = pd.DataFrame(index=df.index, columns=cols, dtype=float)
    for i in range(window, len(df) + 1):
        win = df.iloc[i - window : i]
        X = sm.add_constant(win[cols])
        fit = sm.OLS(win["ret"], X).fit()
        out.iloc[i - 1] = fit.params.loc[cols].values
    return out.dropna()
