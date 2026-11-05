import numpy as np
import streamlit as st
from src.exercises.neural_network import forward_pass, relu, sigmoid, train_network
from src.ui import apply_global_style, exercise_result, interview_questions, math_panel, page_header, section_header, what_to_notice


def make_moons(samples, noise=.18, random_state=42):
    rng = np.random.default_rng(random_state)
    count = (samples + 1) // 2
    angles = np.linspace(0, np.pi, count)
    first = np.column_stack([np.cos(angles), np.sin(angles)])
    second = np.column_stack([1 - np.cos(angles), 1 - np.sin(angles) - .5])
    labels = np.concatenate([np.zeros(count, dtype=int), np.ones(samples-count, dtype=int)])[:samples]
    return np.vstack([first, second])[:samples] + rng.normal(0, noise, np.vstack([first, second])[:samples].shape), labels


def make_circles(samples, noise=.12, factor=.5, random_state=42):
    rng = np.random.default_rng(random_state)
    outer_count = (samples + 1) // 2
    inner_count = samples - outer_count
    outer_angles = rng.random(outer_count) * 2*np.pi
    inner_angles = rng.random(inner_count) * 2*np.pi
    outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner = factor*np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    labels = np.concatenate([np.zeros(outer_count, dtype=int), np.ones(inner_count, dtype=int)])
    X = np.vstack([outer, inner]) + rng.normal(0, noise, (samples, 2))
    return X, labels


apply_global_style(); page_header("Neural Networks","Train a small network and compare its nonlinear decisions with its learning behavior.",eyebrow="DEEP LEARNING")
section_header("1. Concept")
st.markdown("A neural network composes affine transformations with nonlinear activations. Backpropagation efficiently applies the chain rule to compute every parameter gradient.")
math_panel("Two-layer forward pass",r"Z_1=XW_1+b_1,\quad A_1=\operatorname{ReLU}(Z_1),\quad \hat y=\sigma(A_1W_2+b_2)","The hidden nonlinearity prevents stacked layers from collapsing into one linear transformation.")

section_header("2. Activation function reference")
activation_controls,activation_visual=st.columns([1,3],gap="large")
with activation_controls:
    activation=st.selectbox("Activation",["ReLU","sigmoid"],label_visibility="collapsed")
values=np.linspace(-5,5,200)
outputs,done=exercise_result(relu if activation=="ReLU" else sigmoid,values)
if done:
    with activation_visual:
        st.line_chart({"z":values,"Activation":outputs},x="z",y="Activation")
with activation_controls:
    st.caption("ReLU zeros negative inputs. Sigmoid keeps outputs between 0 and 1, but saturates at large magnitudes.")

section_header("3. Interactive Playground")
control_columns=st.columns(6,gap="medium")
with control_columns[0]: shape=st.selectbox("Data shape",["Moons","Circles"])
with control_columns[1]: hidden_units=st.slider("Hidden units",1,32,8)
with control_columns[2]: learning_rate=st.slider("Learning rate",.0001,1.,.05,.0001,format="%.4f")
with control_columns[3]: iterations=st.slider("Iterations",100,2000,500,100)
with control_columns[4]: samples=st.slider("Samples",100,500,300,50)
with control_columns[5]: seed=st.number_input("Random seed",0,999,42)

noise=.18 if shape=="Moons" else .12
if shape=="Moons":
    X,y=make_moons(samples=samples,noise=noise,random_state=int(seed))
else:
    X,y=make_circles(samples=samples,noise=noise,factor=.5,random_state=int(seed))

parameters,loss_history=train_network(X,y,hidden_units,learning_rate,iterations,seed=int(seed))
probabilities,_=forward_pass(X,parameters)
predictions=(probabilities.ravel()>=.5).astype(int)
accuracy=float(np.mean(predictions==y))

plot_columns=st.columns(2,gap="large")
with plot_columns[0]:
    st.scatter_chart({"x1":X[:,0],"x2":X[:,1],"class":y},x="x1",y="x2",color="class")
with plot_columns[1]:
    st.line_chart({"iteration":np.arange(len(loss_history)),"loss":loss_history},x="iteration",y="loss")
metric_columns=st.columns(3)
metric_columns[0].metric("Training accuracy",f"{accuracy:.1%}")
metric_columns[1].metric("Final loss",f"{loss_history[-1]:.4f}")
metric_columns[2].metric("Architecture",f"2 → {hidden_units} → 1")
what_to_notice("Increasing hidden units can create more flexible nonlinear boundaries, while a large learning rate can make training unstable or diverge.")

section_header("4. Interview Prep")
interview_questions([
{"question":"Why do nonlinear activations matter?","answer":"Without them, stacked linear layers collapse to one linear transformation and cannot represent nonlinear decision boundaries."},
{"question":"What is backpropagation?","answer":"An efficient reverse-mode chain-rule procedure that reuses intermediate derivatives to compute the loss gradient for every parameter."},
{"question":"Why can sigmoid cause vanishing gradients?","answer":"At large magnitudes it saturates near zero or one, where its derivative is tiny; repeated multiplication through layers shrinks gradients."},
{"question":"ReLU strengths and weaknesses?","answer":"It is cheap and has a non-saturating positive-side gradient, but negative inputs have zero gradient and units can become permanently inactive."},
{"question":"Why randomize weight initialization?","answer":"Identical initialization preserves symmetry, causing units in a layer to learn identical features. Scale also matters for stable activation variance."},
{"question":"What does a learning rate control?","answer":"Parameter update magnitude. Too small is slow, but too large can overshoot, destabilize training, or diverge."},
{"question":"Training loss falls but validation loss rises—why?","answer":"The network is overfitting: improving fit to training-specific patterns while generalization degrades."},
])
