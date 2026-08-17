import streamlit as st
from src.ui import apply_global_style, page_header, section_header, todo_box, interview_box

st.set_page_config(page_title="Model Evaluation | ML Interview Lab", page_icon="🧠", layout="wide")
apply_global_style()

page_header("Model Evaluation", "Practice choosing metrics based on the decision problem—not habit.", eyebrow="EVALUATION")

positive_rate = st.slider("Positive class prevalence", 0.01, 0.50, 0.10, 0.01)
threshold = st.slider("Classification threshold", 0.0, 1.0, 0.50, 0.01)

todo_box("Create synthetic predicted probabilities. As threshold changes, update TP, FP, TN, FN, precision, recall, specificity, and F1.")

# TODO:
# - Build the confusion matrix from scratch.
# - Plot ROC and precision-recall curves.
# - Compare what happens as the positive class becomes rare.
# - Explain why ROC AUC can look strong when PR AUC is weak.

section_header("Cross-validation")
todo_box("Add a second example comparing ordinary K-fold, stratified K-fold, and time-series splits. Use a dataset where choosing the wrong split causes leakage.")
interview_box("Suppose a fraud classifier has 99% accuracy. What additional information do you need before deciding whether that is good?")
