# Determining Which Glassdoor Sub-Ratings Have the Strongest Associations with a Company's Overall Rating

***Christopher Keesee, MPS-IO***

![Tests](https://github.com/ckeesee02-stack/glassdoor-subratings-analysis/actions/workflows/tests.yml/badge.svg)

## Summary

- Six Glassdoor sub-ratings explain **93.4%** of the variance in Overall ratings across **495 S&P 500 companies** (OLS, 6-month median ratings, May to October 2026).
- **Culture & Values, Senior Management and Career Opportunities** form a clear top tier, outranking the other three sub-ratings in 99.8% of 1,000 bootstrap resamples.
- **Compensation & Benefits** ranks only 5th in dominance, but it has by far the largest unique contribution (unique R² = .028).

## Identification of Research Goal

Determining the percentage variation of Overall Glassdoor Ratings among S&P 500 companies that can be explained by Glassdoor's subratings, and seeing which sub-ratings are associated the most with the overall rating

## Analytical Approach

Overall Glassdoor target values are continuous, along with the sub-ratings. Therefore, the selected analytical approach for this research is the Ordinary Least Squares Multiple Linear Regression approach. This approach is used to determine the coefficient of determination, along with how each sub-rating relates to the Overall Glassdoor value when controlling for the other sub-ratings. A Dominance Analysis is also used to see how much each sub-rating contributes to the overall coefficient of determination.

## Data Requirements & Data Collection

The feature variables consist of the sub-ratings, so all 6 sub-ratings will be included in the model: 1) Compensation & Benefits 2) Work/Life Balance 3) Culture & Values 4) Senior Management 5) Career Opportunities 6) Diversity & Inclusion. The features and overall target were collected by finding the median value of each variable from May to October of 2026 (495 companies).

## Data Understanding

Various tables were utilized to obtain a better understanding of the data that would be used for linear modeling:

**Inter-Correlation Matrix**

| Category | Overall | Compensation & Benefits | Work/Life Balance | Culture | Senior Management | Career Opportunities | Diversity & Inclusion |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **Overall Rating** | 1 | .71 | .77 | .93 | .91 | .87 | .81 |
| **Compensation & Benefits** | .71 | 1 | .43 | .57 | .57 | .62 | .47 |
| **Work/Life Balance** | .77 | .43 | 1 | .83 | .73 | .60 | .74 |
| **Culture & Values** | .93 | .57 | .83 | 1 | .92 | .83 | .88 |
| **Senior Management** | .91 | .57 | .73 | .92 | 1 | .86 | .79 |
| **Career Opportunities** | .87 | .62 | .60 | .83 | .86 | 1 | .71 |
| **Diversity & Inclusion** | .81 | .47 | .74 | .88 | .79 | .71 | 1 |

**Summary of Glassdoor Ratings (1 to 5 Stars)**

| Category | Lower Whisker | Q1 | Median | Q3 | Upper Whisker | IQR |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **Overall Rating** | 2.90 | 3.51 | 3.72 | 3.92 | 4.46 | 0.41 |
| **Diversity & Inclusion** | 3.05 | 3.65 | 3.87 | 4.06 | 4.52 | 0.40 |
| **Compensation & Benefits** | 2.79 | 3.45 | 3.70 | 3.92 | 4.59 | 0.47 |
| **Work/Life Balance** | 2.60 | 3.35 | 3.63 | 3.86 | 4.48 | 0.51 |
| **Culture & Values** | 2.61 | 3.35 | 3.61 | 3.88 | 4.68 | 0.53 |
| **Career Opportunities** | 2.62 | 3.25 | 3.46 | 3.68 | 4.31 | 0.44 |
| **Senior Management** | 2.31 | 3.02 | 3.24 | 3.50 | 4.22 | 0.48 |

Note: Whiskers are the most extreme values within 1.5 × IQR of the quartiles; values beyond them are outliers.

## Data Preparation

Of the 503 S&P 500 tickers (as of 9/30/26), 5 had no Glassdoor rating trend data (HONA, Q, TPL, TKO, VICI) and 3 duplicate share classes (GOOG, FOX, NWS) were excluded, leaving 495 companies. No missing values had to be imputed for the remaining companies.

## Modeling

Variance Inflation Factors were checked to assess multicollinearity. Afterwards, the overall model coefficient of determination was collected and a Dominance Analysis was performed to see how much each variable contributes to the overall coefficient. Dominance weights were bootstrapped (1,000 resamples) to assess rank stability, and each sub-rating's unique R² (the drop in R² when it is removed from the full model) was calculated. An 80-20 train-test split (random_state = 42) was then conducted to split the dataset into a dataset for training the model and a smaller dataset for testing the model.

**Variance Inflation Factors**

| Predictor | VIF |
|:-:|:-:|
| Culture & Values | 14.61 |
| Senior Management | 7.91 |
| Career Opportunities | 4.54 |
| Diversity & Inclusion | 4.46 |
| Work/Life Balance | 3.41 |
| Comp & Benefits | 1.66 |

Note: Culture & Values and Senior Management have VIFs > 5, indicating collinearity. Therefore, their individual regression coefficients are less stable. A Dominance Analysis was selected as the primary basis for interpreting variable importance as a result.

**Overall Results**

| Concept | Values |
|:-:|:-:|
| Sample Size | 495 |
| Coefficient of Determination | 0.934 |
| Root Mean Square Error | 0.081 |
| Mean Absolute Error | 0.061 |

**Dominance Analysis**

| Rank | Subrating | Share of R² | Share of all Overall Rating variance |
|:-:|:-:|:-:|:-:|
| 1 | Culture & Values | 21.4% | 20.0% |
| 2 | Senior Management | 20.4% | 19.1% |
| 3 | Career Opportunities | 18.2% | 17.0% |
| 4 | Diversity & Inclusion | 13.8% | 12.9% |
| 5 | Compensation & Benefits | 13.2% | 12.3% |
| 6 | Work/Life Balance | 13.0% | 12.1% |
| | **Total** | **100%** | **93.4% (R² 0.934)** |

**Training Results**

| Concept | Values |
|:-:|:-:|
| Sample Size | 396 (80% of overall dataset) |
| Coefficient of Determination | 0.938 |
| Root Mean Square Error | 0.081 |
| Mean Absolute Error | 0.061 |

## Evaluation

After obtaining the training results, the predictive validity of the model had to be determined. This was done through evaluating the test set of 99 companies, performing the Leave-One-Out Cross Validation procedure, and performing the 10-fold Cross Validation procedure (10 folds × 5 repeats).

**Testing & Cross-Validation Results**

| Method | R² | RMSE | MAE |
|:-:|:-:|:-:|:-:|
| Test Set (99 companies) | 0.910 | 0.083 | 0.064 |
| Leave-One-Out Cross Validation | 0.930 | 0.083 | 0.062 |
| 10-fold Cross Validation | 0.927 | 0.082 | 0.062 |

## Takeaways

This was a diagnostic research design fundamentally rather than a predictive one. The 90%+ coefficient of determination in the training, test, and cross-validation sets is partly explained by each reviewer providing the Overall rating and all sub-ratings at the same time. Therefore, the ratings could be aligned with the general impression of the company (halo effect). The goal was to identify which sub-ratings are the most strongly associated with the variance in scores. Culture & Values within an organization are the most associated with the variance in overall scores; whereas, the bottom three sub-ratings (Work/Life Balance, Compensation & Benefits, and Diversity & Inclusion, 13.0% to 13.8%) are statistically indistinguishable. The top three outranked the bottom three in 99.8% of 1,000 bootstrap resamples, so they form a clear tier. However, Culture & Values ranked above Senior Management in 89% of resamples, so the #1 position is not completely decisive. Compensation & Benefits had by far the largest unique contribution (unique R² = .028), while Diversity & Inclusion added almost none once all the variables were controlled for (unique R² < .001). These results align very closely with previous industry research (Stansell, 2019; Johnson, 2021).

**Comparison with Industry Literature (Share of R²)**

| Factor / Subrating | Stansell (2019) | Johnson (2021) | My Results |
|:-:|:-:|:-:|:-:|
| Culture & Values | 22% | 19.1% | 21.4% |
| Senior Leadership / Management | 21% | 18.0% | 20.4% |
| Career Opportunities | 19% | 15.6% | 18.2% |
| Diversity & Inclusion | N/A | 11.3% | 13.8% |
| Comp & Benefits | 12% | 10.8% | 13.2% |
| Work/Life Balance | 13% | N/A | 13.0% |

\*Stansell and Johnson used Shapley value decomposition, which produces the same weights as general dominance analysis (Grömping, 2007). Their analyses used individual reviews rather than company medians, and Stansell (2019) also included Business Outlook (14%), so shares are not one-to-one comparisons.

In a follow-up 2023 study, Johnson found that culture & values, senior leadership quality, and career opportunities sustained the strongest associations, even through the Covid-19 Pandemic (Johnson, 2023). This pattern also extends to business outcomes. Toxic culture was the strongest predictor of company attrition and was 10 times more important than compensation (Sull et al., 2022). Although my research tracks Glassdoor Ratings, rather than operational outcomes, the same pattern appears in this data. Companies ranked in the Top Quartile of Culture & Values with Bottom Quartile Pay (n=5) had a median Overall Rating of 3.91. Bottom quartile companies with upper quartile pay (n=11) had a median of 3.48. The samples are limited for each group (and one bottom-quartile company, Pinnacle West, has only 19 reviews), but this illustrates the strength of culture & values in practice.

| Group | Sample Size (n) | Median Overall Rating |
|:-:|:-:|:-:|
| Top Quartile Culture & Values with Bottom Quartile Pay | 5 | 3.91 |
| Bottom Quartile Culture & Values with Upper Quartile Pay | 11 | 3.48 |

This research serves as an Industrial/Organizational foundation regarding factors that are associated with employee sentiment, and how strong each factor is associated with employee sentiment. The purpose of this research is for companies to use these results as a reference when making decisions on how to sustain a healthy workforce. My own research does not link the results to attrition outcomes; however, adjacent work such as the Sull et al. (2022) research, establishes that link.

## Limitations

These results are correlational and do not establish causation. Glassdoor reviewers self-select, so ratings may overrepresent employees with strong positive or negative experiences (self-selection bias). The sample is limited to S&P 500 companies, so results may not generalize to smaller or public-sector employers. October 2026 ratings were month-to-date at the time of collection.

## Repository Structure

```
├── run_all.py                  # Runs the full analysis end to end
├── src/
│   ├── config.py               # Paths, variable names, seeds
│   ├── data.py                 # Loads/validates data, rebuilds the 80/20 split
│   ├── analysis.py             # Descriptives, VIF, OLS, dominance analysis, bootstrap, validation
│   └── charts.py               # Optional charts (written to figures/, not tracked)
├── data/
│   ├── README.md               # Data dictionary, collection method, exclusions
│   └── test_set_tickers.csv    # The 99 held-out test companies
├── results/                    # Every table reported in this README (CSV/JSON)
├── tests/                      # Unit tests (run on synthetic data)
└── requirements.txt
```

## How to Reproduce

1. Place `glassdoor_sp500_medians.csv` in `data/` (see `data/README.md`; the ratings file is not published).
2. Install dependencies and run:

```bash
pip install -r requirements.txt
python run_all.py          # full run, about a minute (1,000 bootstrap resamples)
python run_all.py --quick  # skips the bootstrap
pytest                     # unit tests; no data file needed
```

| Result in this README | File |
|---|---|
| Inter-correlation matrix | `results/correlations.csv` |
| Summary of ratings (quartiles, whiskers, IQR) | `results/descriptives.csv` |
| Variance Inflation Factors | `results/vif.csv` |
| Overall results (R², RMSE, MAE) | `results/model_summary.json` |
| Dominance analysis, unique R², bootstrap intervals | `results/dominance_general.csv` |
| Bootstrap rank tests (99.8%, 89%) | `results/model_summary.json` |
| Complete/conditional dominance | `results/dominance_pairs.csv` |
| Training, test, LOOCV, 10-fold results | `results/validation.csv` |
| Culture vs. pay group medians | `results/culture_pay_groups.csv` |

## Citations

Grömping, U. (2007). Estimators of relative importance in linear regression based on variance decomposition. *The American Statistician, 61*(2), 139–147. https://doi.org/10.1198/000313007X188252

Johnson, R. (2021, November 17). *The pandemic hasn't changed what matters most to employees*. Glassdoor. https://www.glassdoor.com/blog/the-pandemic-hasnt-changed-what-matters-most-to-employees/

Johnson, R. (2023, June 27). *What makes or breaks employee satisfaction remains intact*. Glassdoor. https://www.glassdoor.com/blog/satisfaction-drivers-remain-intact/

Stansell, A. (2019, July 11). *Which workplace factors drive employee satisfaction around the world?* Glassdoor. https://www.glassdoor.com/blog/employee-satisfaction-drivers/

Sull, D., Sull, C., & Zweig, B. (2022, January 11). Toxic culture is driving the Great Resignation. *MIT Sloan Management Review*. https://sloanreview.mit.edu/article/toxic-culture-is-driving-the-great-resignation/

## Tools

Python (pandas, NumPy, scikit-learn, statsmodels, matplotlib). Data collection and analysis code were developed with AI assistance (Claude). The research question, study design, and interpretation of the results are my own.

## License

MIT. See `LICENSE`.
