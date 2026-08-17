import streamlit as st
from src.ui import apply_global_style, page_header, section_header, todo_box, interview_box

st.set_page_config(page_title="Gradient Descent | ML Interview Lab", page_icon="🧠", layout="wide")
apply_global_style()

import numpy as np
import matplotlib.pyplot as plt

page_header("Gradient Descent Playground", "Build gradient descent yourself, then make its behavior visible.", eyebrow="OPTIMIZATION")

section_header("What you should be able to explain")
st.markdown("""
- what a gradient represents
- why the negative gradient is a descent direction
- how learning rate affects convergence
- why feature scaling can matter
- batch vs stochastic vs mini-batch gradient descent
""")

left, right = st.columns([1, 2])
with left:
    learning_rate = st.slider("Learning rate", 0.001, 1.0, 0.10, 0.001)
    initial_x = st.slider("Initial x", -8.0, 8.0, 6.0, 0.1)
    iterations = st.slider("Iterations", 1, 100, 25)
with right:
    st.markdown("Use **f(x) = x² + 2x + 1** for the first version.")
    todo_box("Implement gradient descent. Store every x visited so the path can be plotted. Experiment with learning rates that converge, oscillate, and diverge.")

# TODO:
# 1. Write f(x).
# 2. Write df_dx(x) by hand.
# 3. Implement x <- x - learning_rate * gradient.
# 4. Store x_history and loss_history.
# 5. Overlay optimization steps on this chart.

fig, ax = plt.subplots()
x = np.linspace(-8, 8, 200)
ax.plot(x, x**2 + 2*x + 1)
ax.set_title("TODO: overlay your gradient descent path")
ax.set_xlabel("x")
ax.set_ylabel("Loss")
st.pyplot(fig)

section_header("Stretch challenge")
st.markdown("Extend this to a **2-parameter linear regression loss surface** and visualize how feature scaling changes the optimization path.")
interview_box("If gradient descent moves downhill locally, why can a learning rate that is too large still make the loss increase?")
