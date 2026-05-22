import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier


# =========================================================
# 1. PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="AI Customer Support Ticket Analyzer",
    layout="wide"
)

st.title("AI Customer Support Ticket Analyzer")

st.write(
    "This app uses XGBoost to analyze customer support tickets, "
    "predict ticket type, predict priority, detect sentiment, detect urgency, "
    "and generate a customer support response draft."
)


# =========================================================
# 2. DATASET PATH
# =========================================================

DATASET_PATH = "customer_support_tickets.csv"

st.caption(f"Dataset path: {os.path.abspath(DATASET_PATH)}")


# =========================================================
# 3. LOAD DATASET
# =========================================================

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("")
        else:
            df[col] = df[col].fillna(0)

    return df


if os.path.exists(DATASET_PATH):
    df = load_data(DATASET_PATH)
else:
    st.error(
        "Dataset file not found. Put customer_support_tickets.csv "
        "in the same folder as app.py."
    )
    st.stop()


# =========================================================
# 4. REQUIRED COLUMNS CHECK
# =========================================================

required_columns = [
    "Ticket Subject",
    "Ticket Description",
    "Ticket Type",
    "Ticket Priority",
    "Ticket Status",
    "Customer Satisfaction Rating"
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()


# =========================================================
# 5. TEXT CLEANING
# =========================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================================================
# 6. PREPARE FEATURES FOR XGBOOST
# =========================================================

def prepare_features(df):
    temp_df = df.copy()

    # Main text feature
    temp_df["ticket_text"] = (
        temp_df["Ticket Subject"].fillna("").astype(str) + " " +
        temp_df["Ticket Description"].fillna("").astype(str)
    )

    temp_df["ticket_text"] = temp_df["ticket_text"].apply(clean_text)

    # Categorical features
    categorical_features = []

    possible_categorical_features = [
        "Product Purchased",
        "Ticket Channel",
        "Customer Gender"
    ]

    for col in possible_categorical_features:
        if col in temp_df.columns:
            temp_df[col] = temp_df[col].fillna("Unknown").astype(str)
            categorical_features.append(col)

    # Numeric features
    numeric_features = []

    possible_numeric_features = [
        "Customer Age"
    ]

    for col in possible_numeric_features:
        if col in temp_df.columns:
            temp_df[col] = pd.to_numeric(
                temp_df[col],
                errors="coerce"
            ).fillna(0)

            numeric_features.append(col)

    feature_columns = ["ticket_text"] + categorical_features + numeric_features

    return temp_df, feature_columns, categorical_features, numeric_features


# =========================================================
# 7. TRAIN XGBOOST CLASSIFIER
# =========================================================

@st.cache_resource
def train_xgboost_classifier(df, target_column):
    model_df, feature_columns, categorical_features, numeric_features = prepare_features(df)

    model_df = model_df[feature_columns + [target_column]].dropna()

    X = model_df[feature_columns]
    y = model_df[target_column].astype(str)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    transformers = []

    transformers.append(
        (
            "text",
            TfidfVectorizer(
                stop_words="english",
                max_features=30000,
                ngram_range=(1, 3),
                min_df=2,
                sublinear_tf=True
            ),
            "ticket_text"
        )
    )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        )

    if numeric_features:
        transformers.append(
            (
                "numeric",
                StandardScaler(),
                numeric_features
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    y_pred_encoded = model.predict(X_test)

    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)

    accuracy = accuracy_score(y_test_labels, y_pred_labels)

    report = classification_report(
        y_test_labels,
        y_pred_labels,
        zero_division=0
    )

    labels = list(label_encoder.classes_)

    cm = confusion_matrix(
        y_test_labels,
        y_pred_labels,
        labels=labels
    )

    return {
        "model": model,
        "label_encoder": label_encoder,
        "accuracy": accuracy,
        "report": report,
        "cm": cm,
        "labels": labels,
        "feature_columns": feature_columns,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features
    }


# =========================================================
# 8. SENTIMENT DETECTION
# =========================================================

def detect_sentiment(text):
    text = clean_text(text)

    negative_words = [
        "not working",
        "broken",
        "angry",
        "bad",
        "poor",
        "terrible",
        "worst",
        "frustrated",
        "disappointed",
        "hate",
        "issue",
        "problem",
        "error",
        "failed",
        "failure",
        "refund",
        "delay",
        "unhappy",
        "complaint",
        "crash",
        "bug",
        "unable",
        "cannot",
        "cancel",
        "charged",
        "overcharged"
    ]

    positive_words = [
        "thank",
        "thanks",
        "good",
        "great",
        "excellent",
        "happy",
        "satisfied",
        "love",
        "helpful",
        "resolved",
        "appreciate",
        "perfect",
        "amazing"
    ]

    negative_score = sum(1 for word in negative_words if word in text)
    positive_score = sum(1 for word in positive_words if word in text)

    if negative_score > positive_score:
        return "Negative"
    elif positive_score > negative_score:
        return "Positive"
    else:
        return "Neutral"


# =========================================================
# 9. URGENCY DETECTION
# =========================================================

def detect_urgency(text, predicted_priority):
    text = clean_text(text)
    priority = str(predicted_priority).lower()

    urgent_words = [
        "urgent",
        "asap",
        "immediately",
        "right now",
        "critical",
        "emergency",
        "cannot access",
        "not working",
        "failed",
        "crashed",
        "data loss",
        "refund",
        "angry",
        "deadline",
        "blocked"
    ]

    if priority in ["urgent", "critical", "high"]:
        return "High"

    if priority == "medium":
        return "Medium"

    if any(word in text for word in urgent_words):
        return "High"

    return "Normal"


# =========================================================
# 10. RESPONSE GENERATION
# =========================================================

def generate_response(ticket_type, priority, sentiment):
    ticket_type_lower = str(ticket_type).lower()
    priority_lower = str(priority).lower()
    sentiment_lower = str(sentiment).lower()

    if "technical" in ticket_type_lower:
        response = (
            "Hi, thank you for reporting this technical issue. "
            "I understand this may be affecting your ability to use the product. "
            "Our support team will review the issue and guide you through the next steps."
        )

    elif "billing" in ticket_type_lower or "payment" in ticket_type_lower:
        response = (
            "Hi, thank you for reaching out about your billing concern. "
            "We will review the payment or invoice details and help resolve the issue as quickly as possible."
        )

    elif "refund" in ticket_type_lower:
        response = (
            "Hi, thank you for contacting us. "
            "We understand you are requesting a refund. "
            "Please share the order details so our team can review your request."
        )

    elif "product" in ticket_type_lower:
        response = (
            "Hi, thank you for your product inquiry. "
            "We will review your question and provide the most relevant information shortly."
        )

    elif "cancellation" in ticket_type_lower or "cancel" in ticket_type_lower:
        response = (
            "Hi, thank you for contacting us. "
            "We understand you need help with cancellation. "
            "Our team will review your request and assist you with the next steps."
        )

    else:
        response = (
            "Hi, thank you for contacting customer support. "
            "We have received your request and will review it shortly."
        )

    if priority_lower in ["urgent", "critical", "high"] or sentiment_lower == "negative":
        response += (
            " We understand this may be urgent, so we will prioritize your ticket."
        )

    return response


# =========================================================
# 11. DASHBOARD HELPER FUNCTIONS
# =========================================================

def find_column(df, possible_names):
    for col in possible_names:
        if col in df.columns:
            return col
    return None


def shorten_label(label):
    label = str(label)

    replacements = {
        "Billing inquiry": "Billing",
        "Cancellation request": "Cancel",
        "Product inquiry": "Product",
        "Refund request": "Refund",
        "Technical issue": "Technical",
        "High": "High",
        "Low": "Low",
        "Medium": "Medium",
        "Urgent": "Urgent"
    }

    return replacements.get(label, label)


def plot_confusion_matrix(cm, labels, title):
    short_labels = [shorten_label(label) for label in labels]

    fig, ax = plt.subplots(figsize=(4.8, 3.8), dpi=160)

    im = ax.imshow(
        cm,
        cmap="Blues",
        aspect="equal"
    )

    ax.set_title(title, fontsize=9, pad=8)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Actual", fontsize=8)

    ax.set_xticks(np.arange(len(short_labels)))
    ax.set_yticks(np.arange(len(short_labels)))

    ax.set_xticklabels(
        short_labels,
        rotation=35,
        ha="right",
        fontsize=6
    )

    ax.set_yticklabels(
        short_labels,
        fontsize=6
    )

    max_value = cm.max() if cm.max() != 0 else 1

    for i in range(len(short_labels)):
        for j in range(len(short_labels)):
            value = cm[i, j]
            text_color = "white" if value > max_value / 2 else "black"

            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=6
            )

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.035,
        pad=0.03
    )

    cbar.ax.tick_params(labelsize=6)

    plt.tight_layout()

    return fig


# =========================================================
# 12. DATASET PREVIEW
# =========================================================

st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

st.subheader("Dataset Shape")

shape_col1, shape_col2 = st.columns(2)

with shape_col1:
    st.metric("Rows", df.shape[0])

with shape_col2:
    st.metric("Columns", df.shape[1])


# =========================================================
# 13. DATASET QUALITY CHECK
# =========================================================

st.header("Dataset Quality Check")

quality_df = df.copy()

quality_df["clean_ticket_text"] = (
    quality_df["Ticket Subject"].fillna("").astype(str) + " " +
    quality_df["Ticket Description"].fillna("").astype(str)
).apply(clean_text)

type_conflicts = (
    quality_df
    .groupby("clean_ticket_text")["Ticket Type"]
    .nunique()
    .reset_index()
)

type_conflicts = type_conflicts[type_conflicts["Ticket Type"] > 1]

priority_conflicts = (
    quality_df
    .groupby("clean_ticket_text")["Ticket Priority"]
    .nunique()
    .reset_index()
)

priority_conflicts = priority_conflicts[priority_conflicts["Ticket Priority"] > 1]

quality_col1, quality_col2 = st.columns(2)

with quality_col1:
    st.metric("Text Conflicts for Ticket Type", type_conflicts.shape[0])

with quality_col2:
    st.metric("Text Conflicts for Priority", priority_conflicts.shape[0])

with st.expander("View Dataset Conflict Examples"):
    st.write("Ticket Type Text Conflicts")
    st.dataframe(type_conflicts.head(20), use_container_width=True)

    st.write("Ticket Priority Text Conflicts")
    st.dataframe(priority_conflicts.head(20), use_container_width=True)


# =========================================================
# 14. TRAIN XGBOOST MODELS
# =========================================================

with st.spinner("Training XGBoost ticket type model..."):
    type_result = train_xgboost_classifier(df, "Ticket Type")

with st.spinner("Training XGBoost ticket priority model..."):
    priority_result = train_xgboost_classifier(df, "Ticket Priority")

type_model = type_result["model"]
priority_model = priority_result["model"]

type_encoder = type_result["label_encoder"]
priority_encoder = priority_result["label_encoder"]


# =========================================================
# 15. MODEL PERFORMANCE
# =========================================================

st.header("Model Performance")

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric(
        "Ticket Type Accuracy",
        f"{type_result['accuracy']:.2%}"
    )

with metric_col2:
    st.metric(
        "Priority Accuracy",
        f"{priority_result['accuracy']:.2%}"
    )

with metric_col3:
    avg_rating = df["Customer Satisfaction Rating"].mean()

    # Convert 1–5 rating scale to percentage
    avg_satisfaction_percentage = (avg_rating / 5) * 100

    st.metric(
        "Avg Satisfaction",
        f"{avg_satisfaction_percentage:.2f}%"
    )


# =========================================================
# 16. ADVANCED DASHBOARD GRAPHS
# =========================================================

st.header("Advanced Ticket Analytics Dashboard")

category_col = find_column(df, [
    "Ticket Type",
    "category",
    "Category",
    "ticket_category"
])

priority_col = find_column(df, [
    "Ticket Priority",
    "priority",
    "Priority"
])

satisfaction_col = find_column(df, [
    "Customer Satisfaction Rating",
    "customer_satisfaction_score",
    "satisfaction_score",
    "Satisfaction Score"
])

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

fig.suptitle(
    "Customer Support Ticket Analytics",
    fontsize=18,
    fontweight="bold"
)


# Graph 1: Satisfaction Distribution
if satisfaction_col:
    satisfaction_data = df[satisfaction_col].dropna()

    axes[0, 0].hist(
        satisfaction_data,
        bins=20,
        color="#7BC8C4",
        edgecolor="black",
        alpha=0.8
    )

    satisfaction_counts = satisfaction_data.value_counts().sort_index()

    axes[0, 0].plot(
        satisfaction_counts.index,
        satisfaction_counts.values,
        color="#008B8B",
        linewidth=2,
        marker="o"
    )

    axes[0, 0].set_title("Distribution of Customer Satisfaction Score")
    axes[0, 0].set_xlabel("Score")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(True, alpha=0.3)

else:
    axes[0, 0].text(
        0.5,
        0.5,
        "Satisfaction column not found",
        ha="center",
        va="center"
    )


# Graph 2: Ticket Count by Category
if category_col:
    category_counts = df[category_col].value_counts().head(10)

    category_colors = [
        "#3B1F5C",
        "#443C78",
        "#316B83",
        "#238A8D",
        "#1F9E89",
        "#35B779",
        "#6CCE59",
        "#B4DE2C",
        "#FDE725",
        "#F8961E"
    ]

    axes[0, 1].barh(
        category_counts.index,
        category_counts.values,
        color=category_colors[:len(category_counts)]
    )

    axes[0, 1].set_title("Ticket Count by Category")
    axes[0, 1].set_xlabel("Count")
    axes[0, 1].set_ylabel("Category")
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(axis="x", alpha=0.3)

else:
    axes[0, 1].text(
        0.5,
        0.5,
        "Category column not found",
        ha="center",
        va="center"
    )


# Graph 3: Tickets by Priority Level
if priority_col:
    priority_counts = df[priority_col].value_counts()

    priority_colors = [
        "#3B1F5C",
        "#843B84",
        "#CC5A71",
        "#F4A261",
        "#E9C46A"
    ]

    axes[1, 0].bar(
        priority_counts.index,
        priority_counts.values,
        color=priority_colors[:len(priority_counts)]
    )

    axes[1, 0].set_title("Tickets by Priority Level")
    axes[1, 0].set_xlabel("Priority")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].grid(axis="y", alpha=0.3)

else:
    axes[1, 0].text(
        0.5,
        0.5,
        "Priority column not found",
        ha="center",
        va="center"
    )


# Graph 4: Correlation Heatmap
numeric_df = df.select_dtypes(include=["int64", "float64"])

if numeric_df.shape[1] >= 2:
    corr = numeric_df.corr()

    im = axes[1, 1].imshow(
        corr,
        cmap="coolwarm",
        aspect="auto",
        vmin=-1,
        vmax=1
    )

    axes[1, 1].set_title("Correlation Heatmap of Numeric Features")

    axes[1, 1].set_xticks(np.arange(len(corr.columns)))
    axes[1, 1].set_yticks(np.arange(len(corr.columns)))

    axes[1, 1].set_xticklabels(
        corr.columns,
        rotation=90,
        fontsize=8
    )

    axes[1, 1].set_yticklabels(
        corr.columns,
        fontsize=8
    )

    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            value = corr.iloc[i, j]
            text_color = "white" if abs(value) > 0.5 else "black"

            axes[1, 1].text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7
            )

    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

else:
    axes[1, 1].text(
        0.5,
        0.5,
        "Not enough numeric columns",
        ha="center",
        va="center"
    )

plt.tight_layout()
st.pyplot(fig)


# =========================================================
# 17. ANALYZE NEW CUSTOMER TICKET
# =========================================================

st.header("Analyze a New Customer Ticket")

user_subject = st.text_input("Ticket Subject")

user_description = st.text_area(
    "Ticket Description",
    height=150
)

input_col1, input_col2 = st.columns(2)

with input_col1:
    user_product = st.text_input("Product Purchased", value="")

with input_col2:
    user_channel = st.selectbox(
        "Ticket Channel",
        ["", "Email", "Phone", "Chat", "Social media"]
    )

input_col3, input_col4 = st.columns(2)

with input_col3:
    user_age = st.text_input("Customer Age", value="")

with input_col4:
    user_gender = st.selectbox(
        "Customer Gender",
        ["", "Male", "Female", "Other"]
    )


if st.button("Analyze Ticket"):

    if user_description.strip() == "":
        st.warning("Please enter a ticket description.")

    else:
        cleaned_text = clean_text(user_subject + " " + user_description)

        input_data = pd.DataFrame([{
            "ticket_text": cleaned_text,
            "Product Purchased": user_product if user_product else "Unknown",
            "Ticket Channel": user_channel if user_channel else "Unknown",
            "Customer Gender": user_gender if user_gender else "Unknown",
            "Customer Age": float(user_age) if str(user_age).strip().isdigit() else 0
        }])

        raw_type_encoded = type_model.predict(input_data)[0]
        raw_priority_encoded = priority_model.predict(input_data)[0]

        predicted_type = type_encoder.inverse_transform([raw_type_encoded])[0]
        predicted_priority = priority_encoder.inverse_transform([raw_priority_encoded])[0]

        sentiment = detect_sentiment(cleaned_text)
        urgency = detect_urgency(cleaned_text, predicted_priority)

        response = generate_response(
            predicted_type,
            predicted_priority,
            sentiment
        )

        st.subheader("Prediction Results")

        result_col1, result_col2, result_col3, result_col4 = st.columns(4)

        with result_col1:
            st.metric("Ticket Type", predicted_type)

        with result_col2:
            st.metric("Priority", predicted_priority)

        with result_col3:
            st.metric("Sentiment", sentiment)

        with result_col4:
            st.metric("Urgency", urgency)

        st.subheader("AI Draft Response")
        st.success(response)


# =========================================================
# 18. MODEL ERROR ANALYSIS
# =========================================================

st.header("Model Error Analysis")

with st.expander("Ticket Type Classification Report"):
    st.text(type_result["report"])

with st.expander("Ticket Priority Classification Report"):
    st.text(priority_result["report"])

with st.expander("Ticket Type Confusion Matrix"):
    cm_col, empty_col = st.columns([1, 2])

    with cm_col:
        fig_type_cm = plot_confusion_matrix(
            type_result["cm"],
            type_result["labels"],
            "Ticket Type Confusion Matrix"
        )

        st.pyplot(
            fig_type_cm,
            use_container_width=False
        )

with st.expander("Ticket Priority Confusion Matrix"):
    cm_col, empty_col = st.columns([1, 2])

    with cm_col:
        fig_priority_cm = plot_confusion_matrix(
            priority_result["cm"],
            priority_result["labels"],
            "Ticket Priority Confusion Matrix"
        )

        st.pyplot(
            fig_priority_cm,
            use_container_width=False
        )