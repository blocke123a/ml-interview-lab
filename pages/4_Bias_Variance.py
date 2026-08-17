import streamlit as st
from src.ui import apply_global_style, page_header, section_header, todo_box, interview_box

st.set_page_config(page_title="Bias Variance | ML Interview Lab", page_icon="🧠", layout="wide")
apply_global_style()

import numpy as np
import matplotlib.pyplot as plt

page_header("Bias–Variance Playground", "See underfitting and overfitting emerge as model complexity changes.", eyebrow="GENERALIZATION")

degree = st.slider("Polynomial degree", 1, 15, 3)
noise = st.slider("Noise level", 0.0, 1.5, 0.3, 0.05)
n_samples = st.slider("Training samples", 10, 200, 40)

todo_box("Generate noisy samples from a known nonlinear function, fit polynomial regression, and show the training fit and generalization behavior.")

# TODO:
# 1. Use a true function such as sin(x).
# 2. Generate training data.
# 3. Fit polynomial regression of the chosen degree.
# 4. Show train and test error.
# 5. Re-sample repeatedly to ESTIMATE bias and variance.
# 6. Plot expected test error vs complexity.

grid = np.linspace(-3, 3, 300)
fig, ax = plt.subplots()
ax.plot(grid, np.sin(grid))
ax.set_title("True function — add your fitted model")
st.pyplot(fig)

interview_box("Why does reducing training error not necessarily improve test performance? Describe the answer using bias and variance.")
