# CreditWise — Loan Approval Insights

CreditWise is a small interactive data science app built with Streamlit. It brings the
analysis from `Credit_Wise.ipynb` into a browser-based dashboard so you can explore the
loan application data, compare classification models, and try an applicant-level estimate.

> **Important:** Predictions are educational estimates from a sample dataset. They are
not financial advice, an actual lender's decision, or a substitute for fair-lending
review. Do not enter real applicants' personally identifying or confidential information.

## What the app does

- **Overview:** Shows dataset size, approval rate, sample records, and outcome counts.
- **Model performance:** Compares Logistic Regression, K-Nearest Neighbors, and Naive
  Bayes using precision, recall, F1 score, accuracy, and a confusion matrix.
- **Applicant estimate:** Accepts applicant attributes and returns the selected model's
  estimated outcome and approval probability.

The models use an 80/20 stratified train/test split (random state 42) for the displayed
metrics. Missing input values are imputed within the model pipeline. The winning model
is selected by precision on the holdout split; models used for estimates are then fit on
all labeled rows.

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit interface, preprocessing, model evaluation, and estimates |
| `loan_approval_data.csv` | Dataset used by the app; keep beside `app.py` |
| `Credit_Wise.ipynb` | Original exploratory analysis and modeling notebook |
| `requirements.txt` | Python package requirements |

## Run locally

Use Python 3.10 or newer. From the project folder, run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit prints a local URL (usually <http://localhost:8501>) to open in your browser.
Run the app with `streamlit run` rather than the VS Code **Run Python File** button.

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) and select **Create app**.
3. Choose the repository and branch, set **Main file path** to `app.py`, and deploy.
4. Keep `app.py`, `requirements.txt`, and `loan_approval_data.csv` at the repository root.

## Notes

- The dataset contains unlabeled rows; these are excluded from model training and scoring.
- Results depend on this dataset and its label distribution and should not be used to make
  real credit decisions.
- Review the dataset's source, license, privacy, and suitability before redistributing it.
