import streamlit as st
from src.ui import concept_card, page_header, section_header

page_header("How Should I Frame This Problem?", "This page details a framework for planning a data science project, which makes it a good starting point for review. This is not the only framework to follow, but it is generally my approach.", "ML INTERVIEW LAB · DECISION MAP")
section_header("1. Do you have a target label?")
supervised, unsupervised = st.columns(2, gap="large")
with supervised.container(border=True):
    st.markdown("## Supervised learning")
    st.write("You have historical inputs paired with an outcome you want to predict.")
    st.markdown("**Numeric target → Regression**")
    st.caption("Price, demand, duration, revenue")
    st.markdown("**Categorical target → Classification**")
    st.caption("Fraud, churn, diagnosis, routing")
    st.markdown("**Future time-indexed target → Forecasting**")
    st.caption("Sales, load, traffic, inventory")
with unsupervised.container(border=True):
    st.markdown("## Unsupervised learning")
    st.write("You have features but no single known outcome label.")
    st.markdown("**Discover groups → Clustering**")
    st.caption("Customer segments, behavior patterns")
    st.markdown("**Compress or visualize → Dimensionality reduction**")
    st.caption("PCA, embeddings, latent structure")
    st.markdown("**Find unusual cases → Anomaly detection**")
    st.caption("Failures, abuse, rare behavior")

section_header("2. Choose the task family")
cards=[
 ("Regression","Predict a continuous value.","Linear → regularized → trees → boosting","pages/10_Regression_Overview.py"),
 ("Classification","Predict a label (or probability).","Logistic → trees → ensembles → neural networks","pages/11_Classification_Overview.py"),
 ("Clustering","Discover meaningful groups.","K-means → hierarchical → DBSCAN → mixtures","pages/12_Clustering_Overview.py"),
 ("Forecasting","Predict future values in time order.","Naive → smoothing → ARIMA → feature models","pages/13_Forecasting_Overview.py")]
for pair in (cards[:2],cards[2:]):
    columns=st.columns(2,gap="large")
    for column,(title,body,footer,page) in zip(columns,pair):
        with column:
            concept_card(title,body,footer)
            st.page_link(page,label="Open roadmap",icon=":material/arrow_forward:")

section_header("3. Choose models only after defining success")
st.markdown("Before model selection, define the prediction unit, information available at prediction time, baseline, metric, cost of each error, etc. Much of this will depend on the end user of these models.")
st.warning("Forecasting is usually supervised, but it gets its own path because temporal order changes feature engineering, validation, leakage, and deployment.")
st.markdown("Not every project is predictive. Bayesian inference, experimentation, causal inference, and descriptive analytics answer different questions and appear as complementary paths.")
