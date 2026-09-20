import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from src.exercises.regularization import polynomial_features
from src.ui import apply_global_style, exercise_result, implementation_guide, interview_questions, math_panel, page_header, section_header

apply_global_style(); page_header("Bias–Variance","See underfitting and overfitting emerge as complexity changes.",eyebrow="GENERALIZATION")
section_header("1. Concept")
st.markdown("Simple models may miss stable structure (high bias); highly flexible models may chase sample noise (high variance) and can be less interpretable. Test error reflects both plus noise.")
math_panel("Expected prediction error",r"\mathbb E[(y-\hat f(x))^2]=\operatorname{Bias}[\hat f(x)]^2+\operatorname{Var}[\hat f(x)]+\sigma^2","Expected squared error separates systematic misspecification, sensitivity to the training sample, and irreducible noise.")
section_header("2. Interactive Playground")
controls, visual=st.columns([1,2],gap="large")
with controls.container(border=True):
    st.markdown("#### Experiment controls")
    degree=st.slider("Polynomial degree",1,15,3); noise=st.slider("Noise level",0.,1.5,.3,.05); n=st.slider("Training samples",10,200,40)
rng=np.random.default_rng(42); x=np.sort(rng.uniform(-3,3,n)); y=np.sin(x)+rng.normal(0,noise,n); design,done=exercise_result(polynomial_features,x,degree)
grid=np.linspace(-3,3,300); fig,ax=plt.subplots(); ax.plot(grid,np.sin(grid),label="True function"); ax.scatter(x,y,alpha=.5,label="Samples")
if done:
    coefficients=np.linalg.lstsq(design,y,rcond=None)[0]; grid_design,ok=exercise_result(polynomial_features,grid,degree)
    if ok: ax.plot(grid,grid_design@coefficients,label="Fit using your features")
ax.set_title("Complexity and generalization"); ax.legend()
with visual.container(border=True):
    st.pyplot(fig, width="stretch")
    st.caption("**What to notice:** Complexity reduces approximation bias but can make the fitted curve sensitive to sample noise.")
section_header("3. Interview Prep")
interview_questions([
{"question":"Why can lower training error worsen test performance?","answer":"Extra flexibility can fit sample-specific noise (overfitting), reducing training error while increasing variance on new samples."},
{"question":"How does more training data usually help?","answer":"It primarily reduces estimation variance, but it does not fix a systematically misspecified model."},
{"question":"Is model bias the same as fairness bias?","answer":"No. Statistical bias here means systematic prediction error across samples, whereas fairness bias concerns disparate behavior or outcomes across groups."},
{"question":"How does regularization affect the tradeoff?","answer":"Stronger regularization restricts flexibility, usually increasing bias and reducing variance."},
{"question":"Can a model have both high bias and high variance?","answer":"Yes. A poorly specified model trained on limited or unstable data can be systematically wrong and sensitive to the sample. You should probably not present this model."},
])
