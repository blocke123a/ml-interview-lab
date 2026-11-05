import numpy as np
import streamlit as st
from src.exercises.bayesian import beta_binomial_update, beta_pdf, posterior_mean
from src.ui import apply_global_style, exercise_result, interview_questions, math_panel, page_header, section_header

apply_global_style(); page_header("Bayesian Updating","Make priors, evidence, and posterior uncertainty tangible.",eyebrow="STATISTICS")
section_header("1. Concept")
st.markdown("Bayes' rule combines prior beliefs with the likelihood of observed data. Beta and Binomial distributions are conjugate, so the posterior remains Beta.")
math_panel("Bayes' rule",r"P(\theta\mid D)=\frac{P(D\mid\theta)P(\theta)}{P(D)}","Posterior belief is proportional to how well the parameter explains the data times its prior plausibility.")
section_header("2. Interactive Playground")
controls, visual=st.columns([1,2],gap="large")
with controls.container(border=True):
    st.markdown("#### Prior and evidence")
    a=st.slider("Prior α",1,30,2); b=st.slider("Prior β",1,30,2); s=st.slider("Successes",0,100,7); f=st.slider("Failures",0,100,3)
posterior,done=exercise_result(beta_binomial_update,a,b,s,f); x=np.linspace(.001,.999,400)
chart={"x":x,"Prior":beta_pdf(x,a,b)}
if done:
    pa,pb=posterior; chart["Posterior"]=beta_pdf(x,pa,pb); mean,ok=exercise_result(posterior_mean,pa,pb)
    if ok: st.metric("Your posterior mean",f"{mean:.3f}")
with visual.container(border=True):
    st.line_chart(chart,x="x",y=[name for name in chart if name != "x"])
    st.caption("**What to notice:** More evidence narrows the posterior; stronger priors resist movement from small samples.")
section_header("3. Interview Prep")
interview_questions([
{"question":"Credible interval vs confidence interval?","answer":"A Bayesian credible interval assigns posterior probability to a parameter range. A frequentist confidence procedure has stated long-run coverage across repeated samples."},
{"question":"What is a conjugate prior?","answer":"A prior whose posterior belongs to the same distribution family after combining with a given likelihood."},
{"question":"How do α and β affect a Beta prior?","answer":"Their ratio controls location and their sum controls concentration."},
{"question":"What happens as data accumulates?","answer":"The likelihood usually dominates a fixed prior and posterior uncertainty contracts."},
{"question":"What is prior–data conflict?","answer":"Observed data strongly disagree with regions favored by the prior."},
{"question":"What is a posterior predictive distribution?","answer":"It averages future predictions over posterior parameter uncertainty."},
{"question":"Is a flat prior always non-informative?","answer":"No; flatness depends on parameterization."},
])
