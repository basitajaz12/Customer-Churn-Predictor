import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

DATA_URL = "https://raw.githubusercontent.com/blastchar/telco-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])  # drop rows with missing TotalCharges
    df = df.drop(columns=["customerID"])
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df

@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    model = LogisticRegression(max_iter=1000)
    clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    return clf, auc, categorical_cols, numeric_cols, X

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

st.title("📉 Customer Churn Predictor")
st.write(
    "Predict the likelihood of a customer leaving using the Telco Customer Churn dataset. "
    "Adjust the inputs on the left to see the churn probability."
)

with st.spinner("Loading data and training model..."):
    df = load_data()
    model, auc, cat_cols, num_cols, X = train_model(df)

st.success(f"Model trained. ROC AUC: {auc:.3f}")

# Build input form
st.sidebar.header("Customer Details")

input_data = {}
for col in X.columns:
    if col in num_cols:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        input_data[col] = st.sidebar.slider(col, min_val, max_val, mean_val)
    else:
        options = sorted(df[col].dropna().unique().tolist())
        input_data[col] = st.sidebar.selectbox(col, options)

input_df = pd.DataFrame([input_data])

proba = model.predict_proba(input_df)[0][1]

st.subheader("Churn Probability")
st.metric("Chance of Churn", f"{proba*100:.1f}%")

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

st.caption("Dataset: IBM Telco Customer Churn (public dataset)")
