# Data

## What is published

| File | Contents |
|---|---|
| `test_set_tickers.csv` | The 99 tickers in the held-out test set of the 80/20 split. Stored so the exact split can be reproduced regardless of row order. |

## What is not published

The company-level ratings file (`glassdoor_sp500_medians.csv`) is not included because it is derived from Glassdoor's site content. It is listed in `.gitignore`. To rerun the analysis, place it in this folder with one row per company and these columns:

| Column | Description | Required |
|---|---|---|
| `ticker` | S&P 500 ticker | Yes |
| `company` | Company name | No |
| `sector` | GICS sector | No |
| `reviews` | Glassdoor review count | No |
| `Overall` | Overall rating | Yes |
| `Comp` | Compensation & Benefits | Yes |
| `WLB` | Work/Life Balance | Yes |
| `Culture` | Culture & Values | Yes |
| `SeniorMgmt` | Senior Management | Yes |
| `Career` | Career Opportunities | Yes |
| `DI` | Diversity & Inclusion | Yes |

## How the values were collected

- **Source:** each company's Glassdoor "Company ratings over time" chart, collected September 30 to October 1, 2026.
- **Window:** monthly values from May to October 2026. October was month-to-date at collection.
- **Value used:** the median of the six monthly values, for the Overall rating and each sub-rating (1 to 5 scale).

## Sample

- **Starting list:** 503 S&P 500 tickers as of 9/30/26.
- **No rating trend data (5):** HONA, Q, TPL, TKO, VICI.
- **Duplicate share classes dropped (3):** GOOG, FOX, NWS (same employers as GOOGL, FOXA, NWSA).
- **Final sample:** 495 companies, with no missing values.

## Note on row order

The bootstrap and repeated 10-fold cross-validation use fixed seeds, so their exact output depends on the CSV's row order. Re-sorting the file changes those results in the third decimal place. The train/test split, full-sample model, dominance weights and LOOCV are unaffected by row order.
