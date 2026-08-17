import streamlit as st
from src.ui import apply_global_style, page_header, section_header, todo_box, interview_box

st.set_page_config(page_title="Linear Regression | ML Interview Lab", page_icon="🧠", layout="wide")
apply_global_style()

import numpy as np
import matplotlib.pyplot as plt

page_header("Linear Regression", "Rebuild ordinary least squares from intuition, geometry, and code.", eyebrow="CLASSICAL ML")

seed = st.number_input("Random seed", 0, 9999, 42)
noise = st.slider("Noise", 0.0, 10.0, 2.0, 0.1)
n = st.slider("Number of samples", 20, 500, 100)

rng = np.random.default_rng(seed)
x = rng.uniform(-5, 5, n)
y = 3.0 * x + 2.0 + rng.normal(0, noise, n)

fig, ax = plt.subplots()
ax.scatter(x, y, alpha=0.65)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Synthetic regression data")
st.pyplot(fig)

section_header("Build OLS yourself")
todo_box("Implement slope and intercept from first principles. Then optionally derive β = (XᵀX)⁻¹Xᵀy. Do not start with sklearn.")

# TODO:
# 1. Compute x_bar and y_bar.
# 2. Implement the closed-form slope.
# 3. Implement intercept.
# 4. Generate predictions.
# 5. Overlay the fitted line.
# 6. Compute MSE yourself.
# 7. THEN compare against sklearn.linear_model.LinearRegression.

st.code("def fit_simple_linear_regression(x, y):\n    # TODO: return intercept, slope\n    ...", language="python")

section_header("Assumptions")
st.markdown("Add interactive examples for **nonlinearity, heteroskedasticity, outliers, and correlated residuals**.")
interview_box("What assumptions are required for OLS coefficients to be unbiased? Which extra assumptions are needed for standard hypothesis tests and confidence intervals?")
