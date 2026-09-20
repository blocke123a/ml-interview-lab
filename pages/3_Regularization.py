import numpy as np
import streamlit as st
from src.exercises.regularization import l1_penalty, l2_penalty
from src.ui import apply_global_style, exercise_result, implementation_guide, interview_questions, math_panel, page_header, section_header

apply_global_style(); page_header("Ridge vs Lasso","Explore how coefficient penalties trade bias for stability.",eyebrow="REGULARIZATION")
section_header("1. Concept")
st.markdown("Ridge adds an L2 penalty and smoothly shrinks coefficients. Lasso adds an L1 penalty and can produce exact zeros. Scale features before comparing penalties.")
left_math,right_math=st.columns(2)
with left_math: math_panel("Ridge objective",r"\operatorname{MSE}+\alpha\sum_j\beta_j^2","L2 penalizes squared coefficient magnitude.")
with right_math: math_panel("Lasso objective",r"\operatorname{MSE}+\alpha\sum_j|\beta_j|","L1 has corners that can drive coefficients exactly to zero.")
section_header("2. Interactive Playground")
controls, visual=st.columns([1,2],gap="large")
with controls.container(border=True):
    st.markdown("#### Penalty controls")
    size=st.slider("Coefficient vector size",2,30,8); scale=st.slider("Coefficient scale",0.,5.,1.,.1); alpha=st.slider("Regularization strength",0.,20.,1.,.1)
w=np.linspace(-scale,scale,size); l1,a=exercise_result(l1_penalty,w); l2,b=exercise_result(l2_penalty,w)
with visual.container(border=True):
    if a and b:
        m1,m2=st.columns(2); m1.metric("α × L1 penalty",f"{alpha*l1:.3f}"); m2.metric("α × L2 penalty",f"{alpha*l2:.3f}")
    st.bar_chart({"coefficient":w})
    st.caption("**What to notice:** L1 grows linearly; L2 increasingly punishes large individual coefficients.")
section_header("3. Interview Prep")
interview_questions([
{"question":"Why can Lasso select features while Ridge usually cannot?","answer":"L1 constraint regions have corners on coordinate axes, so a loss contour often touches where coefficients are exactly zero. L2 boundaries are smooth."},
{"question":"Why standardize before regularization?","answer":"Penalties act on coefficient magnitude. Without scaling, measurement units determine how strongly each feature is effectively penalized."},
{"question":"How does alpha affect bias and variance?","answer":"Increasing alpha typically raises bias and lowers variance."},
{"question":"What happens with correlated predictors?","answer":"Ridge tends to distribute weight among them. Lasso may select one somewhat arbitrarily, making feature selection unstable."},
{"question":"Should the intercept be penalized?","answer":"Not usually, since it controls baseline level rather than feature sensitivity."},
{"question":"Can regularization improve training error?","answer":"It generally worsens unpenalized training fit, but can improve held-out performance by reducing estimation variance."},
])
