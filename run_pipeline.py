import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(str(Path.cwd()))

from src.config import SEED, setup_plotting_style
from src.data_loader import load_df, find_data_dir
from src.preprocessing import split_dataset, process_features, profile_dataset
from src.models import (
    train_cox_cv, 
    train_rsf_cv, 
    predict_surv_on_grid_cox, 
    predict_surv_on_grid_rsf, 
    bounded_median_time
)
from src.csa import CSAPredictor
from src.evaluation import compute_performance_metrics
from src.visualization import plot_curves, plot_csa_width_hist, plot_individual_prediction

def main():
    setup_plotting_style()
    
    # Setup Output Directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    print("--- 🚀 Starting SigBERT Pipeline ---")
    
    # 1. Load Data
    data_dir = find_data_dir()
    print(f"📂 Loading data from: {data_dir}")
    try:
        df_main = load_df(data_dir / "df_study_L36_w6.csv")
    except FileNotFoundError:
        print("❌ Error: 'df_study_L36_w6.csv' not found in data/.")
        sys.exit(1)
        
    # 2. Preprocessing
    print("🛠️  Preprocessing...")
    df_train, df_calib, df_test = split_dataset(df_main)
    
    from src.data_loader import get_sig_nonconst
    sig_cols, _ = get_sig_nonconst(df_train)
    
    X_train, X_calib, X_test, selected_features = process_features(df_train, df_calib, df_test, sig_cols)
    print(f"   - Features selected: {len(selected_features)}")
    print(f"   - Split sizes: Train={len(df_train)}, Calib={len(df_calib)}, Test={len(df_test)}")

    # 3. Model Training
    print("🧠 Training Models (this may take a moment)...")
    
    # Cox
    print("   - Training CoxPH ElasticNet...")
    cox_results = train_cox_cv(X_train, df_train)
    cox_model = cox_results["model"]
    
    # RSF
    print("   - Training Random Survival Forest...")
    rsf_results = train_rsf_cv(X_train, df_train)
    rsf_model = rsf_results["model"]
    
    # 4. Evaluation
    print("📊 Evaluating Performance...")
    t_min = df_train["time"].min()
    t_max = df_train["time"].max()
    grid = np.linspace(t_min, t_max, 200)

    # Predictions
    surv_cox = predict_surv_on_grid_cox(cox_model, X_test, grid)
    surv_rsf = predict_surv_on_grid_rsf(rsf_model, X_test, grid)
    
    risk_cox = cox_model.predict_partial_hazard(X_test).values.ravel()
    risk_rsf = rsf_model.predict(X_test)

    # Metrics
    perf_cox = compute_performance_metrics(risk_cox, surv_cox, df_train, df_test, grid)
    perf_rsf = compute_performance_metrics(risk_rsf, surv_rsf, df_train, df_test, grid)

    # Save Metrics
    metrics = {
        "Cox": {k: v for k, v in perf_cox.items() if "times" not in k and "scores" not in k},
        "RSF": {k: v for k, v in perf_rsf.items() if "times" not in k and "scores" not in k}
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("   - Metrics saved to results/metrics.json")

    # Plot AUC
    if perf_cox['auc_scores'] is not None:
        plot_curves(
            perf_cox['auc_times'], 
            {"Cox ElasticNet": perf_cox['auc_scores'], "Random Survival Forest": perf_rsf['auc_scores']}, 
            "Time-Dependent AUC Comparison", 
            "AUC(t)",
            output_path=output_dir / "auc_comparison.png"
        )
        print("   - Figure saved: auc_comparison.png")

    # 5. Conformal Prediction (CSA)
    print("🔮 Running Conformal Survival Analysis (CSA)...")
    surv_cox_calib = predict_surv_on_grid_cox(cox_model, X_calib, grid)
    mu_calib = np.array([bounded_median_time(row, grid) for row in surv_cox_calib])
    
    # We use Cox predictions for the CSA base here
    mu_test = np.array([bounded_median_time(row, grid) for row in surv_cox])
    
    csa = CSAPredictor(alpha=0.10)
    csa.fit(df_train, df_calib, mu_calib, grid)
    lower, upper = csa.predict(mu_test, grid)
    
    csa_stats = csa.evaluate(df_test, lower, upper)
    with open(output_dir / "csa_stats.json", "w") as f:
        json.dump(csa_stats, f, indent=4)
        
    # Plot Widths
    plot_csa_width_hist(upper - lower, output_path=output_dir / "csa_widths.png")
    print("   - Figure saved: csa_widths.png")
    
    # Plot Individual
    # Find an event and a censored example
    events = df_test[df_test["event"] == True].index
    censored = df_test[df_test["event"] == False].index
    
    if len(events) > 0:
        idx = 0 # Relative index in test set
        # Need to find the integer location in df_test corresponding to events[0]
        # But df_test is reset_index. 
        # Let's just pick the first few rows
        for i in range(min(5, len(df_test))):
            plot_individual_prediction(
                grid, surv_cox[i], lower[i], upper[i], 
                df_test.iloc[i]["time"], df_test.iloc[i]["event"], 
                f"Patient Prediction (ID {i})",
                output_path=output_dir / f"patient_{i}_prediction.png"
            )
        print("   - Figures saved: Patient predictions")

    print(f"\n✅ Pipeline Complete. Artifacts in {output_dir.absolute()}")

if __name__ == "__main__":
    main()
