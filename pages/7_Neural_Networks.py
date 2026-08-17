import streamlit as st
from src.ui import apply_global_style, page_header, section_header, todo_box, interview_box

st.set_page_config(page_title="Neural Networks | ML Interview Lab", page_icon="🧠", layout="wide")
apply_global_style()

page_header("Neural Networks from Scratch", "Build a tiny network before relying on PyTorch or TensorFlow.", eyebrow="DEEP LEARNING")

hidden_units = st.slider("Hidden units", 1, 32, 4)
activation = st.selectbox("Activation", ["ReLU", "tanh", "sigmoid"])
learning_rate = st.slider("Learning rate", 0.0001, 1.0, 0.05, 0.0001, format="%.4f")

todo_box("Build a 2-layer NumPy neural network for XOR or a two-moons dataset. Implement the forward pass first, then derive backpropagation by hand.")

st.markdown("""
Recommended sequence:

1. initialize weights and biases
2. implement one activation function
3. write the forward pass
4. compute the loss
5. derive gradients
6. write the backward pass
7. train with gradient descent
8. visualize the decision boundary
""")

st.code(
"class TinyNeuralNetwork:\n"
"    def __init__(self, input_dim, hidden_dim, output_dim):\n"
"        ...\n"
"    def forward(self, X):\n"
"        ...\n"
"    def backward(self, X, y, cache):\n"
"        ...\n"
"    def step(self, grads, learning_rate):\n"
"        ...",
language="python",
)

section_header("Stretch goals")
st.markdown("Compare activations, plot hidden activations, demonstrate vanishing gradients, add L2 regularization, then recreate the network in PyTorch.")
interview_box("Why do nonlinear activation functions matter? What happens if every layer uses only linear transformations?")
