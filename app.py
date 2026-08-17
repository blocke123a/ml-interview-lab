import streamlit as st
from src.ui import apply_global_style, concept_card, page_header, section_header

st.set_page_config(page_title="ML Interview Lab", page_icon="🧠", layout="wide")
apply_global_style()

page_header(
    "ML Interview Lab",
    "An interactive study playground for data science and machine learning interviews.",
    eyebrow="BUILD • LEARN • EXPLAIN",
)

st.markdown("""
This app is intentionally built as a **learning project**, not a finished encyclopedia.
The interface and structure are in place, while many mathematical and modeling pieces
are left for you to implement.

The goal is simple: **if you can build the visualization and explain what it means,
you probably understand the concept well enough to discuss it in an interview.**
""")

section_header("Core study tracks")
c1, c2, c3 = st.columns(3)
with c1:
    concept_card("Optimization", "Gradient descent, learning rates, convergence, and loss surfaces.", "Start with Gradient Descent")
    concept_card("Regression", "OLS, regularization, bias–variance, and model assumptions.", "Build from first principles")
with c2:
    concept_card("Classification", "Logistic regression, thresholds, boundaries, and calibration.", "Focus on interpretation")
    concept_card("Bayesian Statistics", "Priors, likelihoods, posteriors, and uncertainty.", "Practice updating beliefs")
with c3:
    concept_card("Neural Networks", "Forward passes, activations, backpropagation, and optimization.", "Build a tiny network")
    concept_card("Model Evaluation", "Cross-validation, ROC/PR, imbalance, and metric tradeoffs.", "Think like an interviewer")

section_header("How to use this repository")
st.markdown("""
**1. Pick one topic.** Open the matching page from the sidebar.

**2. Read the TODOs before writing code.** Each page contains guided implementation prompts rather than finished solutions.

**3. Implement the concept yourself.** Prefer NumPy and simple Python before reaching for a library implementation.

**4. Compare against scikit-learn.** Once your implementation works, use a library model as a reference.

**5. Explain the result in your own words.** Each page includes interview prompts designed to make you articulate the intuition behind the math.
""")

st.info(
    "Suggested rule: use documentation, textbooks, and your own notes first. "
    "Use an LLM mainly for debugging, UI work, or checking an explanation after you have written it."
)

section_header("Recommended first milestone")
st.markdown("Finish **Gradient Descent, Linear Regression, Logistic Regression, and Bayesian Updating** before expanding the app.")
