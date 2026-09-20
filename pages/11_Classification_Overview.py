import streamlit as st
from src.catalog import METHODS
from src.ui import method_catalog_table, section_header, task_header, use_case_panel

task_header("Classification", "Classification Roadmap", "Predict a label or probability, then choose a decision threshold.", "TASK PATH")
use_case_panel(["Target is categorical", "Different errors have different costs", "Probabilities or rankings drive decisions"], ["A numeric magnitude is the true target", "Accuracy is the only stated objective", "Training labels occur after deployment time"], ["Regression", "Clustering", "Anomaly detection"])
section_header("A practical workflow")
st.markdown("1. Define the positive class and error costs.\n2. Establish a majority or rule baseline.\n3. Split data without leakage and preserve groups/time.\n4. Start with logistic regression.\n5. Evaluate discrimination, threshold metrics, and calibration separately.")
section_header("Methods to learn")
method_catalog_table(METHODS["classification"])
st.page_link("pages/5_Logistic_Regression.py", label="Start with Logistic Regression", icon=":material/arrow_forward:")
