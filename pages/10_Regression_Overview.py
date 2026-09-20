import streamlit as st
from src.catalog import METHODS
from src.ui import method_catalog_table, section_header, task_header, use_case_panel

task_header("Regression", "Regression Roadmap", "Predict a continuous outcome and understand what drives it.", "TASK PATH")
use_case_panel(["Target is numeric", "Magnitude of error matters", "You need a strong interpretable baseline"], ["Target is a class", "Observations are future time points", "Relationships are treated as causal without a causal design"], ["Classification", "Forecasting", "Causal inference"])
section_header("A practical workflow")
st.markdown("1. Define the prediction unit and target.\n2. Establish a mean or median baseline.\n3. Choose Mean Absolute Error (MAE) or Mean Squared Error (MSE) from the cost of errors.\n4. Start linear and inspect residuals.\n5. Add regularization or nonlinear models only when diagnostics justify them.")
section_header("Methods to learn")
method_catalog_table(METHODS["regression"])
st.page_link("pages/2_Linear_Regression.py", label="Start with Linear Regression", icon=":material/arrow_forward:")
