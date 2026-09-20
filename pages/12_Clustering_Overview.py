import streamlit as st
from src.catalog import METHODS
from src.ui import math_panel, method_catalog_table, section_header, task_header, use_case_panel

task_header("Clustering", "Clustering Roadmap", "Discover useful structure when no target labels are available.", "UNSUPERVISED")
use_case_panel(["You need exploratory segments","Similarity has domain meaning","Labels are unavailable"],["You already have a prediction target","Clusters will be treated as objective truth","Distance is meaningless without preprocessing"],["Classification","Dimensionality reduction","Anomaly detection"])
section_header("Core mathematics")
math_panel("K-means objective", r"\min_{C_1,\ldots,C_k}\sum_{j=1}^{k}\sum_{x_i\in C_j}\lVert x_i-\mu_j\rVert_2^2", "K-means assigns observations to centroids so within-cluster squared distance is as small as possible.")
section_header("A practical workflow")
st.markdown("1. Define what similarity should mean.\n2. Scale and encode features deliberately.\n3. Compare methods with different definitions of a cluster.\n4. Check stability across samples and parameter choices.\n5. Validate usefulness with domain evidence.")
section_header("Methods")
method_catalog_table(METHODS["clustering"])
st.page_link("pages/14_Clustering_Lab.py",label="Open the clustering exercise lab",icon=":material/arrow_forward:")
