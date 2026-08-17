import streamlit as st
from src.ui import apply_global_style, page_header, section_header, todo_box, interview_box

st.set_page_config(page_title="Logistic Regression | ML Interview Lab", page_icon="🧠", layout="wide")
apply_global_style()

import numpy as np
import matplotlib.pyplot as plt

page_header("Logistic Regression", "Connect linear scores, probabilities, log-odds, and decision boundaries.", eyebrow="CLASSIFICATION")

z = st.slider("Linear score z", -10.0, 10.0, 0.0, 0.1)
todo_box("Implement sigmoid in src/math_utils.py, import it here, display σ(z), then plot the curve and selected point.")

# TODO:
# from src.math_utils import sigmoid
# probability = sigmoid(np.array([z]))[0]
# st.metric("Predicted probability", f"{probability:.3f}")

grid = np.linspace(-10, 10, 300)
fig, ax = plt.subplots()
ax.axvline(0, linestyle="--", alpha=0.5)
ax.set_ylim(-0.05, 1.05)
ax.set_title("TODO: plot your sigmoid implementation")
ax.set_xlabel("z")
ax.set_ylabel("Probability")
st.pyplot(fig)

section_header("Build a 2D classifier")
st.markdown("Implement **z = Xβ + b**, sigmoid probability, binary cross entropy, gradient descent, probability contours, and a threshold-controlled decision boundary.")
threshold = st.slider("Classification threshold", 0.05, 0.95, 0.50, 0.05)
todo_box("Once your model works, make the threshold slider update the confusion matrix and show how precision/recall change.")

interview_box("Why is logistic regression called regression even though it is primarily used for classification?")
