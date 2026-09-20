import numpy as np
import streamlit as st
from src.exercises.evaluation import accuracy, confusion_matrix_from_scratch, f1_score, precision, recall, specificity
from src.ui import apply_global_style, exercise_result, implementation_guide, interview_questions, math_panel, page_header, section_header

apply_global_style(); page_header("Model Evaluation","Choose metrics based on the problem at hand.",eyebrow="EVALUATION")
section_header("1. Concept")
st.markdown("Metrics compress different error costs. Threshold metrics require a decision cutoff; ranking metrics evaluate ordering across cutoffs. Validation design must match deployment.")
math_panel("Core classification metrics",r"\operatorname{Precision}=\frac{TP}{TP+FP},\quad \operatorname{Recall}=\frac{TP}{TP+FN},\quad F_1=\frac{2PR}{P+R}","Precision conditions on predicted positives; recall conditions on actual positives; F1 balances both.")
section_header("2. Interactive Playground")
controls, visual=st.columns([1,2],gap="large")
with controls.container(border=True):
    st.markdown("#### Scenario controls")
    prevalence=st.slider("Positive prevalence",.01,.50,.10,.01); threshold=st.slider("Classification threshold",0.,1.,.50,.01)
rng=np.random.default_rng(42); y=(rng.random(1000)<prevalence).astype(int); probabilities=np.clip(rng.normal(.25+.5*y,.2),0,1); predicted=(probabilities>=threshold).astype(int)
counts,done=exercise_result(confusion_matrix_from_scratch,y,predicted)
with visual.container(border=True):
    if done:
        tn,fp,fn,tp=counts; st.dataframe({"Predicted negative":[tn,fn],"Predicted positive":[fp,tp]},width="stretch")
        columns=st.columns(5)
        for column,(label,fn_metric) in zip(columns,[("Accuracy",accuracy),("Precision",precision),("Recall",recall),("Specificity",specificity),("F1",f1_score)]):
            value,ok=exercise_result(fn_metric,tn,fp,fn,tp)
            if ok: column.metric(label,f"{value:.3f}")
    st.caption("**What to notice:** Lower thresholds usually increase recall while introducing more false positives.")
section_header("3. Interview Prep")
interview_questions([
{"question":"A fraud model has 99% accuracy. Is that good?","answer":"It depends. Check the confusion matrix and precision/recall to assess. Accuracy is not a great standalone metric for classification."},
{"question":"Precision vs recall?","answer":"Precision asks how many flagged cases are truly positive; recall asks how many true positives were found. Choose based on false-positive and false-negative costs."},
{"question":"When is specificity important?","answer":"When correctly rejecting negatives matters, such as avoiding costly interventions for healthy or legitimate cases."},
{"question":"ROC AUC vs PR AUC under imbalance?","answer":"ROC AUC can look strong because abundant true negatives keep false-positive rate small. PR focuses on positive predictions and is often more revealing for rare positives."},
{"question":"What does F1 ignore?","answer":"It ignores true negatives, treats precision and recall symmetrically, and does not evaluate probability calibration or explicit business costs."},
{"question":"Why stratify cross-validation?","answer":"It keeps class proportions more consistent across folds, reducing unstable estimates when classes are imbalanced."},
{"question":"Why is ordinary K-fold wrong for time series?","answer":"It can train on future observations and validate on the past, leaking temporal information. Use forward-chaining or time-aware splits."},
{"question":"Discrimination vs calibration?","answer":"Discrimination measures ranking/separation; calibration measures whether predicted probabilities match observed frequencies. A model can excel at one and fail at the other."},
])
