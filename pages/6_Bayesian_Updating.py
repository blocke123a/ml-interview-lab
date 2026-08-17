import streamlit as st
from src.ui import apply_global_style, page_header, section_header, todo_box, interview_box

st.set_page_config(page_title="Bayesian Updating | ML Interview Lab", page_icon="🧠", layout="wide")
apply_global_style()

import matplotlib.pyplot as plt

page_header("Bayesian Updating", "Make priors, likelihoods, and posteriors tangible.", eyebrow="STATISTICS")

prior_alpha = st.slider("Prior α", 1, 30, 2)
prior_beta = st.slider("Prior β", 1, 30, 2)
successes = st.slider("Observed successes", 0, 100, 7)
failures = st.slider("Observed failures", 0, 100, 3)

todo_box("Derive the posterior parameters for a Beta prior with Binomial observations. Then plot prior and posterior. Use scipy.stats.beta only for the PDF, not to skip the derivation.")

# TODO:
# posterior_alpha = ?
# posterior_beta = ?
# x = ...
# prior_pdf = ...
# posterior_pdf = ...
# Add a credible interval after the basic visualization works.

fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_title("TODO: prior vs posterior")
ax.set_xlabel("Probability of success")
ax.set_ylabel("Density")
st.pyplot(fig)

section_header("Questions worth exploring")
st.markdown("Add examples for **strong vs weak priors, sample size, prior-data conflict, credible intervals, and posterior predictive reasoning**.")
interview_box("A Bayesian credible interval and a frequentist confidence interval can have similar numerical endpoints. What do they mean differently?")
