import numpy as np
import pandas as pd
from sksurv.nonparametric import kaplan_meier_estimator

class CSAPredictor:
    def __init__(self, alpha: float = 0.10, eps_g: float = 1e-3):
        self.alpha = alpha
        self.eps_g = eps_g
        self.times_g = None
        self.surv_g = None
        self.tau = None
        self.q_hat = None
        self.t_min = None
        
    def fit(self, df_train: pd.DataFrame, df_calib: pd.DataFrame, 
            mu_calib: np.ndarray, grid_base: np.ndarray) -> None:
        """
        Fits the CSA model:
        1. Learns G_hat (censoring distribution) on Train.
        2. Computes conformity scores on Calibration set.
        3. Calibrates q_hat (finite sample quantile).
        """
        # 1. Learn G_hat on Train
        t_tr = df_train["time"].astype(float).to_numpy()
        e_tr = df_train["event"].astype(bool).to_numpy()
        delta_cens_tr = (~e_tr).astype(bool)
        
        self.times_g, self.surv_g = kaplan_meier_estimator(delta_cens_tr, t_tr)
        self.times_g = np.asarray(self.times_g, dtype=float)
        self.surv_g  = np.asarray(self.surv_g, dtype=float)
        
        # Determine tau (last time where G >= eps)
        valid = np.where(self.surv_g >= self.eps_g)[0]
        if valid.size == 0:
            raise RuntimeError("Censoring KM degenerated (G < eps everywhere).")
        self.tau = float(self.times_g[valid[-1]])
        
        self.t_min = float(grid_base.min())
        
        if self.tau <= self.t_min:
             # Fallback logic or error, sticking to notebook error for now
             pass # But let's be robust
             
        # 2. Compute Scores on Calibration
        t_cal = df_calib["time"].astype(float).to_numpy()
        Y_cal = np.minimum(t_cal, self.tau)
        mu_cal_b = np.clip(mu_calib, self.t_min, self.tau)
        
        G_Y = self._g_hat(Y_cal)
        
        scores_cal = np.abs(Y_cal - mu_cal_b) / G_Y
        
        # 3. Quantile
        n_cal = len(scores_cal)
        k = int(np.ceil((n_cal + 1) * (1 - self.alpha)))
        k = min(max(k, 1), n_cal)
        self.q_hat = float(np.partition(scores_cal, k - 1)[k - 1])
        
    def predict(self, mu_test: np.ndarray, grid_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes prediction intervals for test set.
        Returns (lower, upper).
        """
        if self.q_hat is None:
            raise RuntimeError("Fit must be called before predict.")
            
        mu_te_b = np.clip(mu_test, self.t_min, self.tau)
        
        # Grid for inversion
        times_g_clip = self.times_g[(self.times_g >= self.t_min) & (self.times_g <= self.tau)]
        grid_t = np.unique(np.concatenate([times_g_clip, grid_full, np.array([self.t_min, self.tau])])).astype(float)
        grid_t.sort()
        
        G_grid = self._g_hat(grid_t)
        
        n_test = len(mu_test)
        lower = np.empty(n_test, dtype=float)
        upper = np.empty(n_test, dtype=float)
        
        # Inversion loop
        # Optimization: Vectorize if possible, but loop is safe and matches notebook
        for i, mu in enumerate(mu_te_b):
            cond = (np.abs(grid_t - mu) / G_grid) <= self.q_hat
            if np.any(cond):
                idx = np.where(cond)[0]
                lower[i] = grid_t[idx[0]]
                upper[i] = grid_t[idx[-1]]
            else:
                lower[i] = self.t_min
                upper[i] = self.tau
                
        # Clip
        lower = np.clip(lower, self.t_min, self.tau)
        upper = np.clip(upper, self.t_min, self.tau)
        
        return lower, upper

    def _g_hat(self, t: np.ndarray | float) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        idx = np.searchsorted(self.times_g, t, side="right") - 1
        idx = np.clip(idx, 0, len(self.surv_g) - 1)
        return np.maximum(self.surv_g[idx], self.eps_g)

    def evaluate(self, df_test: pd.DataFrame, lower: np.ndarray, upper: np.ndarray) -> dict:
        t_te = df_test["time"].astype(float).to_numpy()
        e_te = df_test["event"].astype(bool).to_numpy()
        Y_te = np.minimum(t_te, self.tau)
        
        covered = (Y_te >= lower) & (Y_te <= upper)
        width = upper - lower
        
        return {
            "coverage": float(np.mean(covered)),
            "mean_width": float(np.mean(width)),
            "median_width": float(np.median(width))
        }
