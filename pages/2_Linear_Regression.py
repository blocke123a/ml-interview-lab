import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from src.exercises.linear_regression import fit_simple_ols, mean_squared_error, predict
from src.ui import apply_global_style, exercise_result, implementation_guide, interview_questions, math_panel, page_header, section_header

apply_global_style(); page_header("Linear Regression", "", eyebrow="CLASSICAL ML")
section_header("1. Concept")
st.markdown("Ordinary Least Squares (OLS) chooses coefficients that minimize squared residuals. The slope describes conditional change, while the intercept anchors the line.")
left_math,right_math=st.columns(2)
with left_math: math_panel("Linear model",r"\hat y=\beta_0+\beta_1x","Beta zero is the intercept; beta one is the expected change in y for a one-unit increase in x.")
with right_math: math_panel("OLS coefficients",r"\hat\beta=(X^\top X)^{-1}X^\top y","The normal equation expresses the solution; numerical code usually prefers a solve or pseudoinverse.")
section_header("2. Interactive Playground")
controls, visual=st.columns([1,2],gap="large")
with controls.container(border=True):
    st.markdown("#### Data controls")
    seed=st.number_input("Random seed",0,9999,42); noise=st.slider("Noise",0.,10.,2.,.1); n=st.slider("Samples",20,500,100)
rng=np.random.default_rng(seed); x=rng.uniform(-5,5,n); y=3*x+2+rng.normal(0,noise,n)
fit,done=exercise_result(fit_simple_ols,x,y); fig,ax=plt.subplots(); ax.scatter(x,y,alpha=.6,label="Data")
if done:
    intercept,slope=fit; grid=np.linspace(x.min(),x.max(),100); line,ok=exercise_result(predict,grid,intercept,slope)
    if ok: ax.plot(grid,line,color="tab:orange",label="Your OLS fit")
    train,ok=exercise_result(predict,x,intercept,slope)
    if ok:
        mse,ok=exercise_result(mean_squared_error,y,train)
        if ok: st.metric("Your MSE",f"{mse:.3f}")
ax.set(xlabel="x",ylabel="y",title="Synthetic regression data"); ax.legend()
with visual.container(border=True):
    st.pyplot(fig, width="stretch")
    st.caption("**What to notice:** More noise widens residuals without changing the underlying data-generating slope.")
section_header("3. Interview Prep")
interview_questions([
{"question":"What assumptions does OLS make?","answer":"Unbiased coefficients require linearity in parameters and no perfect multicollinearity. Homoskedastic, independent, normally distributed errors support usual finite-sample inference."},
{"question":"What does a regression coefficient mean?","answer":"Holding included predictors constant, a one-unit feature increase is associated with a coefficient-sized change in conditional mean. It is not automatically causal."},
{"question":"What happens with multicollinearity?","answer":"Strong collinearity inflates variance and makes individual coefficients unstable."},
{"question":"R-squared vs adjusted R-squared?","answer":"R² never falls when features are added. Adjusted R² penalizes added predictors and can fall when they contribute too little."},
{"question":"Confidence interval vs prediction interval?","answer":"A confidence interval targets a mean response; a prediction interval targets a new outcome and is wider because it includes outcome noise."},
{"question":"Must features be normally distributed?","answer":"No. Predictor normality is not an OLS assumption."},
{"question":"When is Ridge preferable to Lasso?","answer":"Often when correlated predictors carry distributed signal and stable prediction matters. Lasso better matches a plausible sparse solution (because it drops non-significant predictors)."},
{"question":"Why are outliers influential?","answer":"Squared loss gives large residuals disproportionate weight, which in turn gives outliers high leverage."},
])
