import streamlit as st
from src.catalog import METHODS
from src.ui import math_panel, method_catalog_table, section_header, task_header, use_case_panel

task_header("Forecasting", "Forecasting Roadmap", "Predict future values while respecting time, horizon, and information availability.", "SUPERVISED · TEMPORAL")
use_case_panel(["Rows are time ordered","Future values are the target","Trend or seasonality matters"],["Random splitting would be acceptable","Time is merely an identifier","The goal is causal impact"],["Regression","Causal inference","Survival analysis"])
section_header("Core mathematics")
math_panel("Simple exponential smoothing", r"\ell_t=\alpha y_t+(1-\alpha)\ell_{t-1},\qquad \hat y_{t+1}=\ell_t", "Alpha controls how quickly the estimated level responds to recent observations.")
section_header("A practical workflow")
st.markdown("1. Define forecast horizon and update cadence.\n2. Plot the series and create naive baselines.\n3. Use rolling or forward validation.\n4. Engineer only features available at prediction time.\n5. Evaluate by horizon and include uncertainty.")
section_header("Methods")
method_catalog_table(METHODS["forecasting"])
st.page_link("pages/15_Forecasting_Lab.py",label="Open the forecasting exercise lab",icon=":material/arrow_forward:")
