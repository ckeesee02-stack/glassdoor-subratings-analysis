"""Load and validate the company-level dataset."""
import pandas as pd

from .config import DATA_FILE, FEATURES, REQUIRED_COLUMNS, TARGET, TEST_SET_FILE


def load_data(path=DATA_FILE) -> pd.DataFrame:
    """Read the CSV and check required columns and value ranges.

    Row order is kept exactly as saved. The seeded steps (bootstrap and
    repeated k-fold CV) depend on row order, so re-sorting the CSV will change
    those results slightly (third decimal place) even though the data are the same.
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. See data/README.md.")
    if df["ticker"].duplicated().any():
        raise ValueError("Duplicate tickers found.")
    ratings = df[[TARGET] + FEATURES]
    if ratings.isna().any().any():
        raise ValueError("Ratings contain missing values.")
    if ((ratings < 1) | (ratings > 5)).any().any():
        raise ValueError("Ratings must be on the 1 to 5 scale.")
    return df.reset_index(drop=True)


def split_masks(df: pd.DataFrame, path=TEST_SET_FILE):
    """Return boolean (train, test) masks for the published 80/20 split.

    The split was originally drawn with train_test_split(test_size=0.2,
    random_state=42). The resulting 99 test tickers are stored in
    data/test_set_tickers.csv so the exact split can be reproduced regardless
    of row order.
    """
    test_tickers = set(pd.read_csv(path)["ticker"])
    test = df["ticker"].isin(test_tickers).to_numpy()
    if test.sum() != len(test_tickers):
        raise ValueError("Some test-set tickers are not in the dataset.")
    return ~test, test
