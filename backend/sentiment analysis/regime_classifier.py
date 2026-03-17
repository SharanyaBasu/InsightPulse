"""
Market Regime Classifier using a Gaussian HMM.

Approach (from LSEG article):
  - Feature: log returns of 7-day MA of SP500 closing price
  - 2 hidden states: bull (low-vol) and bear (high-vol)
  - hmmlearn GaussianHMM, full covariance
  - Rolling retrain every 20 observations

  Sources:

  https://developers.lseg.com/en/article-catalog/article/market-regime-detection
  https://www.researchgate.net/publication/398722347_Machine_Learning-Driven_Market_Regime_Analysis_in_Equity_Markets_A_Gaussian_Hidden_Markov_Model_Approach
"""

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from datetime import datetime, timedelta

# data                                                                       
def _fetch_sp500(lookback_years=5):
    end = datetime.now()
    start = end - timedelta(days=365 * lookback_years)
    df = yf.download("^GSPC", start=start, end=end, interval="1d", progress=False)
    close = df["Close"].squeeze().dropna()
    return close

def _build_features(close):
    """7-day MA log returns — same feature as the LSEG article."""
    ma = close.rolling(7).mean()
    log_ret = np.log(ma / ma.shift(1)).dropna()
    return log_ret.to_frame(name="log_ret_ma7")

# model                                      
def _fit_hmm(X, n_states=2):
    model = GaussianHMM(n_components=n_states, covariance_type="full",
                        n_iter=1000, random_state=42,)
    model.fit(X)
    return model

def _label_states(model):
    """
    human-readable labels by comparing mean returns of each state
    State with higher mean return = bull; lower (or negative) = bear.
    """
    means = model.means_.flatten()
    bull_state = int(np.argmax(means))
    labels = {}
    for i in range(model.n_components):
        labels[i] = "bull" if i == bull_state else "bear"
    return labels

def get_current_regime(lookback_years: int = 5) -> dict:
    """
    fit a 2-state gaussian HMM on SP500 log returns and return the
    current (most recent) regime plus the full labeled history

    Returns:
        {
            "current_regime": "bull" | "bear",
            "current_state":  int,
            "history": pd.Series  (DatetimeIndex -> "bull"/"bear")
        }
    """
    close = _fetch_sp500(lookback_years)
    features = _build_features(close)
    X = features.values

    model = _fit_hmm(X)
    state_seq = model.predict(X)
    labels = _label_states(model)

    history = pd.Series(
        [labels[s] for s in state_seq],
        index=features.index,
        name="regime",
    )

    current_state = int(state_seq[-1])
    current_regime = labels[current_state]

    return {
        "current_regime": current_regime,
        "current_state": current_state,
        "history": history,
    }

# quick tets
if __name__ == "__main__":
    result = get_current_regime()
    print(f"\nCurrent regime : {result['current_regime'].upper()}  (state {result['current_state']})")
    print("\nLast 10 days:")
    print(result["history"].tail(10).to_string())

    counts = result["history"].value_counts()
    print(f"\nRegime distribution over training window:")
    print(counts.to_string())
