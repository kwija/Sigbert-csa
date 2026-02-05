import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from .config import SEED
from .data_loader import get_sig_nonconst

def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into Train (50%), Calibration (30%), and Test (20%).
    Stratified by event.
    """
    n = df.shape[0]
    n_train = int(np.floor(0.50 * n))
    n_calib = int(np.floor(0.30 * n))
    n_test  = n - n_train - n_calib  # Ensures exact sum

    idx = np.arange(n)
    y_strat = df["event"].astype(int).to_numpy()

    # 1) Train vs Rest
    idx_train, idx_rest = train_test_split(
        idx,
        train_size=n_train,
        random_state=SEED,
        shuffle=True,
        stratify=y_strat,
    )

    # 2) Calib vs Test
    y_rest = y_strat[idx_rest]
    idx_calib, idx_test = train_test_split(
        idx_rest,
        train_size=n_calib,
        test_size=n_test,
        random_state=SEED,
        shuffle=True,
        stratify=y_rest,
    )
    
    # Verification
    set_train, set_calib, set_test = set(idx_train), set(idx_calib), set(idx_test)
    assert len(set_train & set_calib) == 0
    assert len(set_train & set_test) == 0
    assert len(set_calib & set_test) == 0

    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_calib = df.iloc[idx_calib].reset_index(drop=True)
    df_test  = df.iloc[idx_test].reset_index(drop=True)

    return df_train, df_calib, df_test

def process_features(df_train, df_calib, df_test, features_all, k_best=200):
    """
    Imputes missing values (Median), Selects top-K features by variance, and Scales (StandardScaler).
    All fitted on Training set.
    """
    # 1. Impute
    imp = SimpleImputer(strategy="median")
    X_tr0 = pd.DataFrame(imp.fit_transform(df_train[features_all]), columns=features_all)
    X_ca0 = pd.DataFrame(imp.transform(df_calib[features_all]), columns=features_all)
    X_te0 = pd.DataFrame(imp.transform(df_test[features_all]),  columns=features_all)

    # 2. Variance Selection
    var = X_tr0.var(axis=0, ddof=0).sort_values(ascending=False)
    keep_cols = list(var.head(min(k_best, len(var))).index)
    
    # 3. Scale
    sc = StandardScaler()
    X_tr = pd.DataFrame(sc.fit_transform(X_tr0[keep_cols]), columns=keep_cols)
    X_ca = pd.DataFrame(sc.transform(X_ca0[keep_cols]), columns=keep_cols)
    X_te = pd.DataFrame(sc.transform(X_te0[keep_cols]), columns=keep_cols)
    
    return X_tr, X_ca, X_te, keep_cols

def profile_dataset(df: pd.DataFrame, name: str) -> dict:
    sig_cols, sig_nonconst = get_sig_nonconst(df)
    time = df["time"].astype(float)
    event = df["event"].astype(bool)

    return {
        "dataset": name,
        "n": int(df.shape[0]),
        "p_sig": int(len(sig_cols)),
        "p_nonconst": int(len(sig_nonconst)),
        "missing_mean": float(df[["time", "event"] + sig_cols].isna().mean().mean()),
        "missing_max": float(df[["time", "event"] + sig_cols].isna().mean().max()),
        "event_rate": float(event.mean()),
        "censor_rate": float((~event).mean()),
        "time_q10": float(time.quantile(0.10)),
        "time_q25": float(time.quantile(0.25)),
        "time_q50": float(time.quantile(0.50)),
        "time_q75": float(time.quantile(0.75)),
        "time_q90": float(time.quantile(0.90)),
    }
