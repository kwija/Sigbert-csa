import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .config import COLORS

def plot_curves(time_grid, curve_dict, title, ylabel, output_path=None):
    """
    Generic plotter for curves (AUC, Brier).
    curve_dict: {"ModelName": values}
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    styles = [("-", "o", COLORS["teal"]), ("--", "s", COLORS["rose"])]
    
    for i, (name, values) in enumerate(curve_dict.items()):
        ls, mk, col = styles[i % len(styles)]
        ax.plot(time_grid, values, color=col, linestyle=ls, linewidth=2, 
                label=name, alpha=0.9)
                
    ax.set_title(title, pad=15)
    ax.set_xlabel("Time (t)")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.show()

def plot_csa_width_hist(widths, output_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(widths, bins=30, color=COLORS["teal"], edgecolor='white', alpha=0.9)
    median = np.median(widths)
    ax.axvline(median, color=COLORS["rose"], linestyle='--', label=f"Median: {median:.1f}")
    
    ax.set_title("CSA Interval Width Distribution", pad=15)
    ax.set_xlabel("Width (days)")
    ax.legend(frameon=False)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.show()

def plot_individual_prediction(grid, surv_func, lower, upper, obs_time, event, title, output_path=None):
    """Plots individual survival curve with CSA interval."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    # Curve
    ax.plot(grid, surv_func, color=COLORS["teal"], lw=2.5, label="Predicted Survival")
    
    # Interval
    ax.axvspan(lower, upper, color=COLORS["fill"], alpha=0.6, label="CSA Interval")
    
    # Event
    ax.axvline(obs_time, color=COLORS["gray"], linestyle=":", alpha=0.7)
    
    if event:
        val_at_obs = surv_func[np.abs(grid - obs_time).argmin()]
        ax.plot(obs_time, val_at_obs, 'o', color=COLORS["rose"], markersize=8, label="Event Observed")
    else:
        val_at_obs = surv_func[np.abs(grid - obs_time).argmin()]
        ax.plot(obs_time, val_at_obs, 'X', color=COLORS["gray"], markersize=8, label="Censored")

    ax.set_title(title, pad=15, color=COLORS["teal"])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.legend(loc="lower left")
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.show()
