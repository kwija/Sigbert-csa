import os
from pathlib import Path
import matplotlib.pyplot as plt

# Global Seed
SEED = 42

# Paths
ROOT = Path.cwd().resolve()
if ROOT.name.lower() in {"notebooks", "notebook"}:
    ROOT = ROOT.parent

# Data Directory Resolution
DATA_DIR_ENV = os.environ.get("DATA_DIR")
OUT_DIR_ENV  = os.environ.get("OUT_DIR")

# Colors
COLORS = {
    "teal": "#1C4E63",       # Couleur principale
    "rose": "#B97878",       # Accent (Médiane/Alertes)
    "gray": "#4A4A4A",       # Texte secondaire
    "fill": "#D1E3EB",       # Intervalle CSA (Soft Blue/Gray)
    "grid": "#E0E0E0",       # Grille discrète
    "line": "#2C3E50"        # Lignes guides
}

def setup_plotting_style():
    """Configures global matplotlib style for professional output."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 0.8,
        'grid.color': COLORS["grid"],
        'grid.linestyle': '--',
        'grid.alpha': 0.6,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
