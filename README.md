# Indie Games Sales Prediction — Baseline ML Project

## Objective
Build a reproducible preprocessing and baseline regression pipeline to predict `CopiesSold` from the Steam games dataset.

Dataset source: [Top 1500 Games on Steam by Revenue (09-09-2024)](https://www.kaggle.com/datasets/alicemtopcu/top-1500-games-on-steam-by-revenue-09-09-2024)

## Files
- `indie_games_sales_prediction.ipynb` — complete Colab-compatible notebook.
- `README.md` — project documentation and methodology.
- `requirements.txt` — core Python dependencies.

The source CSV is not bundled because it must be downloaded from Kaggle and may be subject to the dataset's terms.

## Run in Google Colab
1. Download `indie_games.csv` from Kaggle.
2. Open the notebook in Google Colab.
3. Run the cells from top to bottom and upload the CSV when prompted.
4. Review the printed audit and feature-removal lists before trusting the results.
5. At the end, download the generated CSVs, metrics, and `.joblib` model artifact from the Colab file browser.

## Pipeline and decisions
1. **Audit:** inspect schema, dtypes, missingness, unique counts, duplicate rows, and sample values.
2. **Cleaning:** normalize column names; remove exact duplicate rows; parse numeric target formatting; exclude rows with missing/non-positive `CopiesSold`.
3. **Feature review:** remove obvious IDs/URLs/free-text fields and constant columns only when present. Review the printed high-cardinality and drop lists; do not remove useful columns blindly.
4. **Missing values:** median imputation for numeric predictors; most-frequent imputation for categorical predictors.
5. **Categories:** One-Hot Encoding for nominal predictors. Label Encoding is shown only as a template for a truly ordinal feature with a defined order; arbitrary label codes are not used for nominal features in the regression model.
6. **Scaling:** StandardScaler for numeric features. MinMaxScaler is compared as an alternative. All preprocessing is fitted on training data only.
7. **Outliers:** IQR counts are reported for review. High-sales games are not automatically deleted because they may be legitimate observations.
8. **Model:** Linear Regression predicts `log1p(CopiesSold)`; predictions are transformed back using `expm1` and clipped at zero.
9. **Evaluation:** held-out RMSE, MAE, and R²; actual-versus-predicted plot, residual histogram, and largest-error table.
10. **Feature analysis:** absolute coefficient ranking on the log-target scale.

## Why use a log-transformed target?
Copies sold is often highly right-skewed. Training on `log1p(CopiesSold)` can reduce the dominance of the most extreme values. Reported RMSE and MAE are calculated in the original copies-sold unit after reversing the transformation.

## Leakage prevention
The train-test split occurs before fitting imputers, encoders, and scalers. A scikit-learn `ColumnTransformer` and `Pipeline` fit those transformations only on the training set and reuse them unchanged on the test set. The target is excluded from predictors.

## Important caveats
- The exact columns, missing values, outliers, and model scores depend on the actual CSV uploaded.
- The notebook does not invent evaluation results. Run it against the source file and use the resulting metrics in your submission.
- A random split measures performance on held-out rows from the same dataset; it does not establish future performance or causality.
- Coefficient magnitude is a simple diagnostic, not definitive feature importance. Correlated predictors and one-hot categories can make coefficients unstable.
- Consider a `DummyRegressor` baseline and cross-validation as future improvements.

## Environment
See `requirements.txt`. Google Colab generally includes the main libraries already.
