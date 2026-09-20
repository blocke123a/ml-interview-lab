import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from src.exercises.logistic_regression import sigmoid
from src.ui import apply_global_style, exercise_result, implementation_guide, interview_questions, math_panel, page_header, section_header

apply_global_style(); page_header("Logistic Regression","Connect scores, log-odds, probabilities, and decisions.",eyebrow="CLASSIFICATION")
section_header("1. Concept")
st.markdown("Logistic regression models log-odds as a linear function, then maps the score through sigmoid. A decision threshold converts probabilities into actions.")
left_math,right_math=st.columns(2)
with left_math: math_panel("Sigmoid probability",r"P(y=1\mid x)=\sigma(X\beta+b),\quad \sigma(z)=\frac{1}{1+e^{-z}}","The sigmoid maps any real-valued score into a probability.")
with right_math: math_panel("Binary cross-entropy",r"-\frac1n\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)]","Minimizing BCE is equivalent to maximizing the Bernoulli likelihood.")
section_header("2. Interactive Playground")
controls, visual=st.columns([1,2],gap="large")
with controls.container(border=True):
    st.markdown("#### Decision controls")
    z=st.slider("Linear score z",-10.,10.,0.,.1); threshold=st.slider("Classification threshold",.05,.95,.5,.05)
grid=np.linspace(-10,10,300); probabilities,done=exercise_result(sigmoid,grid); selected,point_done=exercise_result(sigmoid,z)
if point_done: st.metric("Your predicted probability",f"{float(selected):.3f}",delta=f"{'positive' if selected>=threshold else 'negative'} at threshold")
fig,ax=plt.subplots()
if done: ax.plot(grid,probabilities,label="Your sigmoid")
ax.axhline(threshold,ls="--",alpha=.5,label="Threshold"); ax.axvline(0,ls=":",alpha=.5); ax.set(xlabel="z",ylabel="Probability",ylim=(-.05,1.05)); ax.legend()
with visual.container(border=True):
    st.pyplot(fig, width="stretch")
    st.caption("**What to notice:** The threshold changes the decision, not the probability produced by the model.")
section_header("3. Interview Prep")
interview_questions([
{"question":"Why is it called regression?","answer":"It fits a continuous linear model for log-odds, even though probabilities are commonly thresholded for classification."},
{"question":"How do you interpret a coefficient?","answer":"A one-unit feature increase multiplies the odds by exp(coefficient), holding other features fixed."},
{"question":"Why use cross-entropy as the loss function instead of MSE?","answer":"Binary Cross Entropy is the negative Bernoulli log-likelihood and gives a convex objective for ordinary logistic regression with well-behaved probability gradients."},
{"question":"What does changing the threshold do?","answer":"It changes predicted labels and the precision–recall tradeoff, but not the underlying probabilities."},
{"question":"What happens with perfectly separable data?","answer":"Unregularized maximum-likelihood coefficients can grow without bound while probabilities approach zero and one."},
{"question":"Are logistic probabilities always calibrated?","answer":"No. Misspecification, sampling shifts, regularization, and imbalance handling can harm calibration."},
{"question":"How does logistic regression differ from a linear probability model?","answer":"Sigmoid constrains outputs to (0,1) and models linear log-odds; a linear probability model can predict outside the valid range."},
])
