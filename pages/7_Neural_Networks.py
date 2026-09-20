import numpy as np
import streamlit as st
from src.exercises.neural_network import relu, sigmoid
from src.ui import apply_global_style, exercise_result, implementation_guide, interview_questions, math_panel, page_header, section_header

apply_global_style(); page_header("Neural Networks","",eyebrow="DEEP LEARNING")
section_header("1. Concept")
st.markdown("A neural network composes affine transformations with nonlinear activations. Backpropagation efficiently applies the chain rule to compute every parameter gradient.")
math_panel("Two-layer forward pass",r"Z_1=XW_1+b_1,\quad A_1=\operatorname{ReLU}(Z_1),\quad \hat y=\sigma(A_1W_2+b_2)","The hidden nonlinearity prevents stacked layers from collapsing into one linear transformation.")
section_header("2. Interactive Playground")
controls, visual=st.columns([1,2],gap="large")
with controls.container(border=True):
    st.markdown("#### Architecture controls")
    units=st.slider("Hidden units",1,32,4); activation=st.selectbox("Activation",["ReLU","sigmoid"]); rate=st.slider("Learning rate",.0001,1.,.05,.0001,format="%.4f")
values=np.linspace(-5,5,200); outputs,done=exercise_result(relu if activation=="ReLU" else sigmoid,values)
with visual.container(border=True):
    if done: st.line_chart({"input":values,"activation":outputs},x="input",y="activation")
    st.caption("**What to notice:** ReLU preserves positive-side gradients; sigmoid saturates at large magnitudes.")
st.caption(f"Selected architecture: 2 inputs → {units} hidden units → 1 output; learning rate {rate:.4f}.")
section_header("3. Interview Prep")
interview_questions([
{"question":"Why do nonlinear activations matter?","answer":"Without them, stacked linear layers collapse to one linear transformation and cannot represent nonlinear decision boundaries."},
{"question":"What is backpropagation?","answer":"An efficient reverse-mode chain-rule procedure that reuses intermediate derivatives to compute the loss gradient for every parameter."},
{"question":"Why can sigmoid cause vanishing gradients?","answer":"At large magnitudes it saturates near zero or one, where its derivative is tiny; repeated multiplication through layers shrinks gradients."},
{"question":"ReLU strengths and weaknesses?","answer":"It is cheap and has a non-saturating positive-side gradient, but negative inputs have zero gradient and units can become permanently inactive."},
{"question":"Why randomize weight initialization?","answer":"Identical initialization preserves symmetry, causing units in a layer to learn identical features. Scale also matters for stable activation variance."},
{"question":"What does a learning rate control?","answer":"Parameter update magnitude. Too small is slow, but too large can overshoot, destabilize training, or diverge."},
{"question":"Training loss falls but validation loss rises—why?","answer":"The network is overfitting: improving fit to training-specific patterns while generalization degrades."},
])
