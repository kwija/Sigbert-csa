import pandas as pd
from pathlib import Path
import os
from .config import ROOT, DATA_DIR_ENV

def find_data_dir() -> Path:
    """Locates the data directory based on environment or default locations."""
    # 0) Explicit override
    if DATA_DIR_ENV:
        return Path(DATA_DIR_ENV).resolve()

    # 1) Kaggle style (if applicable, though unlikely in this refactor, kept for robustness)
    on_kaggle = Path("/kaggle").exists()
    kaggle_input = Path("/kaggle/input").resolve()
    if on_kaggle:
        cand = (kaggle_input / "df-study").resolve()
        if cand.exists():
            return cand
        if kaggle_input.exists():
            for d in kaggle_input.iterdir():
                if d.is_dir() and (d / "df_study_L18_w6.csv").exists():
                    return d.resolve()

    # 2) Local candidates
    candidates = [
        (ROOT / "data"),
        (ROOT / "data" / "raw"),
        (ROOT / "data" / "processed"),
        ROOT,
    ]
    for d in candidates:
        if (d / "df_study_L18_w6.csv").exists() or (d / "df_study_L36_w6.csv").exists():
            return d.resolve()

    # Default fallback
    return (ROOT / "data").resolve()

DATA_DIR = find_data_dir()

def load_df(path: Path) -> pd.DataFrame:
    """Loads a CSV dataset and performs basic validation."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
        
    df = pd.read_csv(path)
    if "event" not in df.columns or "time" not in df.columns:
        raise ValueError(f"Expected columns missing in {path.name}: needs ['time','event']")
        
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["event"] = df["event"].astype(int).astype(bool)
    df = df.dropna(subset=["time", "event"]).reset_index(drop=True)
    return df

def get_sig_nonconst(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Identifies signal columns and filters out constant ones."""
    sig_cols = [c for c in df.columns if c.startswith("sig_")]
    nunique = df[sig_cols].nunique(dropna=False)
    sig_nonconst = nunique[nunique > 1].index.tolist()
    return sig_cols, sig_nonconst
