import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from src.exercises.forecasting import seasonal_naive_forecast, simple_exponential_smoothing
from src.ui import exercise_result, implementation_guide, interview_questions, math_panel, playground_layout, section_header, task_header, what_to_notice

task_header("Forecasting","Forecasting Lab","Start with temporal baselines before reaching for complex models.","SUPERVISED · TEMPORAL")
section_header("Concept and mathematics")
math_panel("Seasonal-naive baseline",r"\hat y_{T+h}=y_{T+h-m}","For seasonal period m, reuse the observation from the corresponding position in the latest observed cycle.")
section_header("Interactive playground")
controls,visual=playground_layout()
with controls:
    method=st.selectbox("Forecast method",["Seasonal naive","Exponential smoothing"])
    periods=st.slider("History length",48,240,120,12); horizon=st.slider("Forecast horizon",6,36,24,6)
    season=st.slider("Season length",4,24,12); noise=st.slider("Noise",0.0,2.0,.5,.1)
    alpha=st.slider("Smoothing alpha",.05,.95,.3,.05,disabled=method!="Exponential smoothing")
rng=np.random.default_rng(42); t=np.arange(periods); series=10+.035*t+2*np.sin(2*np.pi*t/season)+rng.normal(0,noise,periods); future=np.arange(periods,periods+horizon)
with visual:
    if method=="Seasonal naive":
        result,complete=exercise_result(seasonal_naive_forecast,series,season,horizon); forecast=result if complete else np.resize(series[-season:],horizon)
        notice="This baseline preserves seasonal shape but assumes no changing trend."
    else:
        result,complete=exercise_result(simple_exponential_smoothing,series,alpha)
        if complete: fitted,next_value=result
        else: fitted=pd.Series(series).ewm(alpha=alpha,adjust=False).mean().to_numpy(); next_value=fitted[-1]
        forecast=np.repeat(next_value,horizon); notice="Larger alpha reacts faster to recent changes but follows noise more closely."
    fig,ax=plt.subplots(); ax.plot(t,series,label="Observed"); ax.plot(future,forecast,label="Forecast",color="#f59e0b",linewidth=2); ax.axvline(periods-1,color="gray",ls="--",alpha=.6); ax.legend(); ax.set(xlabel="Time",ylabel="Value",title=method); st.pyplot(fig,width="stretch"); what_to_notice(notice)
section_header("Interview prep")
interview_questions([
 {"question":"Why not randomly split a time series?","answer":"It can train on observations occurring after validation examples, creating future leakage."},
 {"question":"Why is seasonal naive important?","answer":"It is cheap, transparent, and often strong; complex models must beat it under honest validation."},
 {"question":"What determines a forecast horizon?","answer":"The decision lead time: how far ahead the business must act."},
 {"question":"What is rolling-origin validation?","answer":"Train on an initial history, evaluate on the next horizon, move the origin forward, and repeat."},
 {"question":"Point forecast versus interval?","answer":"A point forecast gives one expected value; an interval communicates plausible future outcomes."}])
