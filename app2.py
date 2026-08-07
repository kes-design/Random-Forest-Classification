"""
Glass Classifier - Streamlit web app
=====================================
Upload a "database" Excel file (each sheet = one class, rows = samples,
columns = element concentrations) to train a Random Forest, then upload
an "unknown samples" Excel file to classify each row.

Run with:
    streamlit run app.py

Requires: streamlit, pandas, scikit-learn, matplotlib, openpyxl
Install with:
    pip install streamlit pandas scikit-learn matplotlib openpyxl
"""

import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

st.set_page_config(page_title="Glass Classifier", layout="wide")
st.title("Glass Sample Classifier (Random Forest)")

st.markdown(
    """
Upload a **database file** where each sheet is a class (e.g. `Flatglas`,
`Verpakkingsglas`, `Telefoonglas`) and rows are samples with element
concentrations as columns. Then upload an **unknown samples file** with
the same element columns to classify.
"""
)

# ---------------------------------------------------------------
# Session state so results survive re-runs / interactions
# ---------------------------------------------------------------
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.feature_cols = None
    st.session_state.class_labels = None


# =================================================================
# STEP 1 — Upload database & train
# =================================================================
st.header("1. Train on your database")

db_file = st.file_uploader("Database Excel file (.xlsx)", type=["xlsx"], key="db")

if db_file is not None:
    xls = pd.ExcelFile(db_file)
    sheet_names = xls.sheet_names
    st.write(f"Found {len(sheet_names)} sheets (= classes): {', '.join(sheet_names)}")

    selected_sheets = st.multiselect(
        "Sheets to use as classes", sheet_names, default=sheet_names
    )

    # Peek at the columns of the first selected sheet to offer an ID-column choice
    preview_cols = list(pd.read_excel(xls, sheet_name=selected_sheets[0], nrows=0).columns) if selected_sheets else []
    id_col_choice = st.selectbox(
        "Sample ID column to exclude from training (not a measurement)",
        options=["(none)"] + preview_cols,
        index=1 if preview_cols else 0,
        help="Pick the column that identifies each sample (e.g. Sample_ID), "
             "so it isn't treated as an element measurement. Defaults to the first column.",
    )

    test_size = st.slider("Test set size (for evaluation)", 0.1, 0.4, 0.25, 0.05)
    n_estimators = st.slider("Number of trees (n_estimators)", 100, 1000, 500, 50)

    if st.button("Train model", type="primary"):
        frames = []
        for sheet in selected_sheets:
            df = pd.read_excel(xls, sheet_name=sheet)
            df = df.dropna(how="all")
            df["label"] = sheet
            frames.append(df)

        data = pd.concat(frames, ignore_index=True)

        excluded_cols = {"label"}
        if id_col_choice != "(none)":
            excluded_cols.add(id_col_choice)

        feature_cols = [c for c in data.columns if c not in excluded_cols]
        X = data[feature_cols]
        y = data["label"]

        st.write(f"Using {len(feature_cols)} feature columns: {', '.join(feature_cols)}")

        st.write("Class counts:")
        st.dataframe(y.value_counts().rename("samples"))

        # Evaluation split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        eval_clf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=42, n_jobs=-1
        )
        eval_clf.fit(X_train, y_train)
        y_pred = eval_clf.predict(X_test)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Test accuracy", f"{accuracy_score(y_test, y_pred):.1%}")
            st.text("Classification report:")
            st.text(classification_report(y_test, y_pred))
        with col2:
            labels = sorted(y.unique())
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            st.text("Confusion matrix:")
            st.dataframe(pd.DataFrame(cm, index=labels, columns=labels))

        importances = pd.Series(
            eval_clf.feature_importances_, index=feature_cols
        ).sort_values(ascending=False)
        st.subheader("Feature importances")
        st.bar_chart(importances)

        # Final model trained on ALL data (used for real classification)
        final_clf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=42, n_jobs=-1
        )
        final_clf.fit(X, y)

        st.session_state.model = final_clf
        st.session_state.feature_cols = feature_cols
        st.session_state.class_labels = labels

        st.success(
            "Final model trained on all available data. "
            "You can now classify unknown samples below."
        )

        # Example tree visualization
        st.subheader("Example decision tree (tree 1 of the forest, top 3 levels)")
        fig, ax = plt.subplots(figsize=(16, 8))
        plot_tree(
            final_clf.estimators_[0],
            feature_names=feature_cols,
            class_names=labels,
            filled=True,
            rounded=True,
            max_depth=3,
            fontsize=7,
            ax=ax,
        )
        st.pyplot(fig)
        st.caption(
            "This is just one of many trees in the forest, shown as an example. "
            "The forest's actual prediction is an average across all trees — "
            "see feature importances above for what really drives it."
        )


# =================================================================
# STEP 2 — Upload unknown samples & classify
# =================================================================
st.header("2. Classify unknown samples")

if st.session_state.model is None:
    st.info("Train a model in step 1 first.")
else:
    unknown_file = st.file_uploader(
        "Unknown samples Excel file (.xlsx)", type=["xlsx"], key="unknown"
    )

    if unknown_file is not None:
        unknown_df = pd.read_excel(unknown_file)
        unknown_df = unknown_df.dropna(how="all")

        missing_cols = [
            c for c in st.session_state.feature_cols if c not in unknown_df.columns
        ]
        if missing_cols:
            st.error(
                f"Unknown samples file is missing required columns: {missing_cols}"
            )
        else:
            X_unknown = unknown_df[st.session_state.feature_cols]

            if X_unknown.isna().any().any():
                st.warning(
                    "Some values are missing in the unknown samples file. "
                    "Rows with missing values may give unreliable predictions."
                )

            model = st.session_state.model
            predictions = model.predict(X_unknown)
            probabilities = model.predict_proba(X_unknown)

            results = unknown_df.copy()
            results["Predicted_class"] = predictions
            for i, cls in enumerate(model.classes_):
                results[f"P({cls})"] = probabilities[:, i]

            st.subheader("Results")
            st.dataframe(results)

            # Downloadable Excel
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                results.to_excel(writer, index=False, sheet_name="Predictions")
            buffer.seek(0)

            st.download_button(
                "Download results as Excel",
                data=buffer,
                file_name="classified_samples.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )