import numpy as np
import pandas as pd
from sksurv.util import Surv
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    brier_score,
)

def compute_performance_metrics(risk_scores, surv_funcs, df_train, df_test, times_grid):
    """
    Computes C-index, Integrated Brier Score, and AUC mean.
    surv_funcs: matrix (n_test, n_times) corresponding to times_grid
    """
    y_train = Surv.from_arrays(
        event=df_train["event"].astype(bool).to_numpy(),
        time=df_train["time"].astype(float).to_numpy(),
    )
    y_test = Surv.from_arrays(
        event=df_test["event"].astype(bool).to_numpy(),
        time=df_test["time"].astype(float).to_numpy(),
    )
    
    # 1. C-Index
    c_index = concordance_index_censored(
        df_test["event"].astype(bool), 
        df_test["time"].astype(float), 
        risk_scores
    )[0]
    
    # 2. IBS & Brier(t)
    # Restrict to common support
    t_min = float(max(df_train["time"].min(), df_test["time"].min()))
    t_max = float(min(df_train["time"].max(), df_test["time"].max()))
    
    mask = (times_grid >= t_min) & (times_grid <= t_max)
    times_eval = times_grid[mask]
    surv_eval = surv_funcs[:, mask]
    
    if times_eval.size < 5:
        ibs = np.nan
        brier_scores = None
    else:
        times_brier, brier_scores = brier_score(y_train, y_test, surv_eval, times_eval)
        # Integrate (Trapezoidal)
        ibs = np.trapezoid(brier_scores, times_brier) / (times_brier.max() - times_brier.min())

    # 3. AUC(t)
    # Use a subset of times for stability (avoid endpoints)
    auc_times = times_eval[(times_eval > times_eval.min()) & (times_eval < times_eval.max())]
    if auc_times.size > 100:
         idx = np.linspace(0, auc_times.size - 1, 100).round().astype(int)
         auc_times = auc_times[idx]
         
    if auc_times.size > 0:
        auc_t, auc_mean = cumulative_dynamic_auc(y_train, y_test, risk_scores, auc_times)
    else:
        auc_mean = np.nan
        auc_t = None
        
    return {
        "c_index": c_index,
        "ibs": ibs,
        "auc_mean": auc_mean,
        "brier_times": times_eval if brier_scores is not None else None,
        "brier_scores": brier_scores,
        "auc_times": auc_times if auc_t is not None else None,
        "auc_scores": auc_t
    }
