import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from src.exercises.gradient_descent import gradient_descent, objective_gradient
from src.ui import apply_global_style, exercise_result, implementation_guide, interview_questions, math_panel, page_header, section_header

apply_global_style(); page_header("Gradient Descent", "", eyebrow="OPTIMIZATION")
section_header("1. Concept")
st.markdown("Gradient descent moves parameters opposite the local gradient. Its learning rate controls the speed of convergence, but with a key tradeoff!")
math_panel("Parameter update",r"\theta_{t+1}=\theta_t-\eta\nabla_\theta J(\theta_t)","Theta is the parameter, eta the learning rate, and the gradient is the local direction of steepest increase.")
section_header("2. Interactive Playground")
left, right = st.columns([1, 2])
with left:
    rate = st.slider("Learning rate", .001, 1.0, .10, .001); start = st.slider("Initial x", -8., 8., 6., .1); steps = st.slider("Iterations", 1, 100, 25)
with right: result, done = exercise_result(gradient_descent, objective_gradient, start, rate, steps)
grid = np.linspace(-8, 8, 300); fig, ax = plt.subplots(); ax.plot(grid, (grid + 1) ** 2, label="Reference objective")
if done:
    xs, losses = result; ax.plot(xs, losses, "o--", label="Your optimization path")
ax.set(xlabel="x", ylabel="Loss", title="f(x) = (x + 1)²"); ax.legend()
with right:
    st.pyplot(fig, width="stretch")
    st.caption("**What to notice:** Stable learning rates approach x = −1; oversized steps oscillate or diverge.")
section_header("3. Interview Prep")
interview_questions([
{"question":"Why does the negative gradient point downhill?","answer":"The gradient is the direction of steepest local increase, so its negative gives steepest local decrease under the Euclidean norm.","testing":"Local geometric intuition."},
{"question":"Why can a large learning rate increase loss?","answer":"The direction can be correct while the step overshoots the low-loss region, causing oscillation or divergence."},
{"question":"Batch vs stochastic vs mini-batch gradient descent?","answer":"They use all samples, one sample, or a subset per update, trading gradient accuracy, computational efficiency, and update noise."},
{"question":"Why does feature scaling help?","answer":"Unequal scales create elongated contours, making one learning rate zigzag across steep dimensions and crawl along shallow ones."},
{"question":"Does gradient descent always find a global minimum?","answer":"No. Convex objectives offer stronger guarantees, whereas non-convex objectives can contain local minima or saddles."},
{"question":"How do you detect convergence?","answer":"Monitor loss change, gradient norm, and parameter-update size, with tolerances and a maximum iteration count."},
])
