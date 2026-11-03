from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path(__file__).parent / "loan_approval_data.csv"
TARGET = "Loan_Approved"
ID_COLUMN = "Applicant_ID"
MODEL_NAMES = ("Logistic Regression", "K-Nearest Neighbors", "Naive Bayes")
NUMERIC_LIMITS = {
    "Age": (18.0, 100.0, 1.0),
    "Dependents": (0.0, 20.0, 1.0),
    "Credit_Score": (300.0, 850.0, 1.0),
    "Existing_Loans": (0.0, 50.0, 1.0),
    "DTI_Ratio": (0.0, 1.0, 0.01),
    "Loan_Term": (1.0, 360.0, 1.0),
}

st.set_page_config(
    page_title="CreditWise | Loan Insights",
    page_icon="💳",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH.name}")
    return pd.read_csv(DATA_PATH)


def build_pipeline(
    estimator: object, numeric_columns: list[str], categorical_columns: list[str]
) -> Pipeline:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )
    return Pipeline([("preprocessing", preprocessing), ("model", estimator)])


@st.cache_resource(show_spinner="Training and evaluating models...")
def train_models(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Pipeline], str, dict[str, pd.DataFrame]]:
    labeled = data.loc[data[TARGET].notna()].drop(columns=[ID_COLUMN], errors="ignore")
    features = labeled.drop(columns=[TARGET])
    target = labeled[TARGET].astype(str)
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = features.select_dtypes(exclude=["number"]).columns.tolist()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
    }

    metrics: list[dict[str, float | str]] = []
    fitted_models: dict[str, Pipeline] = {}
    confusion_matrices: dict[str, pd.DataFrame] = {}
    for name, estimator in estimators.items():
        pipeline = build_pipeline(estimator, numeric_columns, categorical_columns)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metrics.append(
            {
                "Model": name,
                "Precision": precision_score(
                    y_test, predictions, pos_label="Yes", zero_division=0
                ),
                "Recall": recall_score(
                    y_test, predictions, pos_label="Yes", zero_division=0
                ),
                "F1 score": f1_score(
                    y_test, predictions, pos_label="Yes", zero_division=0
                ),
                "Accuracy": accuracy_score(y_test, predictions),
            }
        )
        matrix = confusion_matrix(y_test, predictions, labels=["No", "Yes"])
        confusion_matrices[name] = pd.DataFrame(
            matrix,
            index=["Actual No", "Actual Yes"],
            columns=["Predicted No", "Predicted Yes"],
        )
        fitted_models[name] = build_pipeline(
            estimators[name], numeric_columns, categorical_columns
        ).fit(features, target)

    metrics_frame = pd.DataFrame(metrics).sort_values(
        ["Precision", "F1 score", "Accuracy"], ascending=False
    )
    best_model = str(metrics_frame.iloc[0]["Model"])
    return metrics_frame, fitted_models, best_model, confusion_matrices[best_model]


def numeric_input(column: str, data: pd.DataFrame) -> float:
    values = pd.to_numeric(data[column], errors="coerce").dropna()
    default = float(values.median()) if not values.empty else 0.0
    minimum, maximum, step = NUMERIC_LIMITS.get(
        column, (0.0, max(1.0, float(values.max()) * 2 if not values.empty else 1.0), 1.0)
    )
    default = min(max(default, minimum), maximum)
    return st.number_input(
        column.replace("_", " "),
        min_value=minimum,
        max_value=maximum,
        value=default,
        step=step,
        key=f"input_{column}",
    )


def render_overview(data: pd.DataFrame) -> None:
    labeled = data.loc[data[TARGET].notna()]
    approval_rate = (labeled[TARGET].astype(str) == "Yes").mean() * 100
    columns = st.columns(4)
    columns[0].metric("Applicants", f"{len(data):,}")
    columns[1].metric("Labeled records", f"{len(labeled):,}")
    columns[2].metric("Approval rate", f"{approval_rate:.1f}%")
    columns[3].metric("Input features", f"{data.shape[1] - 2}")

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Approval outcomes")
        counts = labeled[TARGET].astype(str).value_counts().reindex(["Yes", "No"], fill_value=0)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(counts.index, counts.values, color=["#13a89e", "#425466"])
        ax.set_ylabel("Applicants")
        ax.set_xlabel("")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    with right:
        st.subheader("Dataset preview")
        st.dataframe(data.head(10), width="stretch", hide_index=True)
        missing = data.isna().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            st.caption(
                "Missing values are imputed using training data. "
                f"{int(missing.sum())} missing cells across {len(missing)} columns."
            )


def render_performance(
    metrics: pd.DataFrame, best_model: str, best_confusion_matrix: pd.DataFrame
) -> None:
    st.subheader("Held-out test performance")
    st.caption(
        "Metrics use an 80/20 stratified split (random state 42). "
        "The model used for predictions is retrained on all labeled records."
    )
    st.dataframe(
        metrics.style.format(
            {"Precision": "{:.1%}", "Recall": "{:.1%}", "F1 score": "{:.1%}", "Accuracy": "{:.1%}"}
        ),
        width="stretch",
        hide_index=True,
    )
    st.info(f"Highest precision on the test split: **{best_model}**.")
    st.subheader(f"{best_model} confusion matrix")
    st.dataframe(best_confusion_matrix, width="stretch")


def render_prediction(data: pd.DataFrame, models: dict[str, Pipeline], best_model: str) -> None:
    st.write("Enter applicant details to get an estimate from a model trained on the dataset.")
    model_choice = st.selectbox("Prediction model", ["Best precision model", *MODEL_NAMES])
    model_name = best_model if model_choice == "Best precision model" else model_choice

    features = data.drop(columns=[TARGET, ID_COLUMN], errors="ignore")
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = features.select_dtypes(exclude=["number"]).columns.tolist()
    values: dict[str, float | str] = {}
    numeric_form, categorical_form = st.columns(2)
    with numeric_form:
        st.markdown("#### Financial and credit details")
        for column in numeric_columns:
            values[column] = numeric_input(column, data)
    with categorical_form:
        st.markdown("#### Applicant profile")
        for column in categorical_columns:
            categories = sorted(data[column].dropna().astype(str).unique().tolist())
            if not categories:
                st.error(f"No valid choices are available for {column}.")
                return
            values[column] = st.selectbox(
                column.replace("_", " "),
                categories,
                key=f"input_{column}",
            )

    if st.button("Estimate approval", type="primary"):
        applicant = pd.DataFrame([values], columns=features.columns)
        model = models[model_name]
        prediction = str(model.predict(applicant)[0])
        probability = model.predict_proba(applicant)[0]
        approval_index = list(model.classes_).index("Yes")
        confidence = float(probability[approval_index])
        if prediction == "Yes":
            st.success(f"Estimated outcome: **Likely approved** ({confidence:.1%} estimated approval probability)")
        else:
            st.warning(f"Estimated outcome: **Likely not approved** ({confidence:.1%} estimated approval probability)")
        st.caption(
            f"Model: {model_name}. This educational estimate is not a lending decision "
            "or financial advice."
        )


def main() -> None:
    st.title("CreditWise")
    st.markdown("### Loan approval insights")
    st.write(
        "Explore the loan dataset, compare classification models, and estimate "
        "an applicant's approval outcome."
    )
    try:
        data = load_data()
        metrics, models, best_model, best_confusion_matrix = train_models(data)
    except (FileNotFoundError, KeyError, ValueError) as error:
        st.error(f"Unable to load or prepare the loan dataset: {error}")
        st.stop()

    overview_tab, performance_tab, prediction_tab = st.tabs(
        ["Overview", "Model performance", "Applicant estimate"]
    )
    with overview_tab:
        render_overview(data)
    with performance_tab:
        render_performance(metrics, best_model, best_confusion_matrix)
    with prediction_tab:
        render_prediction(data, models, best_model)


if __name__ == "__main__":
    main()
