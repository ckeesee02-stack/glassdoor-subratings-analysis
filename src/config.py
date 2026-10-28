"""Project-wide settings: file paths, variable names, and random seeds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "glassdoor_sp500_medians.csv"
TEST_SET_FILE = ROOT / "data" / "test_set_tickers.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

TARGET = "Overall"
FEATURES = ["Comp", "WLB", "Culture", "SeniorMgmt", "Career", "DI"]

LABELS = {
    "Overall": "Overall Rating",
    "Comp": "Compensation & Benefits",
    "WLB": "Work/Life Balance",
    "Culture": "Culture & Values",
    "SeniorMgmt": "Senior Management",
    "Career": "Career Opportunities",
    "DI": "Diversity & Inclusion",
}

REQUIRED_COLUMNS = ["ticker", TARGET] + FEATURES

# Reproducibility
SEED = 42
TEST_SIZE = 0.20
N_BOOTSTRAP = 1000
CV_FOLDS = 10
CV_REPEATS = 5
