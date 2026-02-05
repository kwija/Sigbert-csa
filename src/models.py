import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.exceptions import ConvergenceError
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from .config import SEED

def train_cox_cv(X_train: pd.DataFrame, df_train_target: pd.DataFrame) -> dict:
    """
    Fast training for demo (No CV).
    """
    y_tr = df_train_target["event"].astype(int).to_numpy()
    t_tr = df_train_target["time"].astype(float).to_numpy()
    
    final_train = X_train.copy()
    final_train["time"] = t_tr
    final_train["event"] = y_tr
    
    # Simple fit, no grid search
    cox_final = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)
    cox_final.fit(final_train, duration_col="time", event_col="event")
    
    return {
        "model": cox_final,
        "best_params": {"penalizer": 0.1, "l1_ratio": 0.5},
        "cv_results": pd.DataFrame()
    }

def train_rsf_cv(X_train: pd.DataFrame, df_train_target: pd.DataFrame) -> dict:
    """
    Fast RSF for demo.
    """
    X = X_train.to_numpy(dtype=np.float32)
    y = Surv.from_arrays(
        event=df_train_target["event"].astype(bool).to_numpy(),
        time=df_train_target["time"].astype(float).to_numpy(),
    )
    
    # Fast parameters
    rsf = RandomSurvivalForest(
        n_estimators=50,  # fast
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=SEED
    )
    rsf.fit(X, y)
    
    return {
        "model": rsf,
        "best_params": {"n_estimators": 50},
        "cv_results": pd.DataFrame()
    }

def predict_surv_on_grid_cox(cph, X_df: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Returns survival matrix (n_samples, n_grid)."""
    sf = cph.predict_survival_function(X_df, times=grid)
    return sf.values.T

def predict_surv_on_grid_rsf(rsf, X_df: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Returns survival matrix (n_samples, n_grid)."""
    # RSF predict_survival_function returns array of StepFunction
    X = X_df.to_numpy(dtype=np.float32) if isinstance(X_df, pd.DataFrame) else X_df
    surv_funcs = rsf.predict_survival_function(X)
    
    mat = np.vstack([fn(grid) for fn in surv_funcs])
    return mat

def bounded_median_time(sf_row: np.ndarray, grid: np.ndarray) -> float:
    """Calculates median time from survival function row."""
    if sf_row[0] <= 0.5:
        return float(grid[0])
    if sf_row[-1] > 0.5:
        return float(grid[-1])
    # first point where S(t) <= 0.5
    idx = int(np.argmax(sf_row <= 0.5))
    return float(grid[idx])
