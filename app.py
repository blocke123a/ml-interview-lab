import streamlit as st
from src.ui import apply_global_style

st.set_page_config(page_title="ML Interview Lab", page_icon="🧠", layout="wide")
apply_global_style()
navigation = st.navigation({
 "Start": [st.Page("app_pages/00_Start_Here.py", title="Choose Your Path", icon=":material/explore:")],
 "Foundations": [st.Page("app_pages/1_Gradient_Descent.py", title="Gradient Descent"), st.Page("app_pages/4_Bias_Variance.py", title="Bias–Variance")],
 "Regression": [st.Page("app_pages/10_Regression_Overview.py", title="Regression Roadmap", icon=":material/trending_up:"), st.Page("app_pages/2_Linear_Regression.py", title="Linear Regression"), st.Page("app_pages/3_Regularization.py", title="Regularization")],
 "Classification": [st.Page("app_pages/11_Classification_Overview.py", title="Classification Roadmap", icon=":material/category:"), st.Page("app_pages/5_Logistic_Regression.py", title="Logistic Regression"), st.Page("app_pages/8_Model_Evaluation.py", title="Classification Evaluation")],
 "Unsupervised": [st.Page("app_pages/12_Clustering_Overview.py", title="Clustering Roadmap", icon=":material/hub:"), st.Page("app_pages/14_Clustering_Lab.py", title="Clustering Lab")],
 "Forecasting": [st.Page("app_pages/13_Forecasting_Overview.py", title="Forecasting Roadmap", icon=":material/timeline:"), st.Page("app_pages/15_Forecasting_Lab.py", title="Forecasting Lab")],
 "Inference & advanced": [st.Page("app_pages/6_Bayesian_Updating.py", title="Bayesian Updating"), st.Page("app_pages/7_Neural_Networks.py", title="Neural Networks")],
 "Production": [st.Page("app_pages/16_Deployment.py", title="Deployment & Monitoring", icon=":material/rocket_launch:")],
})
navigation.run()
