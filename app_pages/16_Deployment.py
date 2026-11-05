import streamlit as st
from src.ui import page_header, section_header, use_case_panel

page_header("Deployment & Monitoring","Choose a serving pattern from the decision context, then design for safe releases, observability, and rollback.","PRODUCTION ML · VENDOR NEUTRAL")
section_header("1. Choose the prediction pattern")
use_case_panel(["Batch: predictions on a schedule","Real time: a response is needed now","Streaming: react continuously to events"],["Latency is assumed rather than measured","Training features are unavailable online","Predictions have no operational owner"],["Human-in-the-loop workflow","Rules or SQL scoring","Analytics dashboard"])
with st.container(border=True):
    st.markdown("#### Interactive deployment guide")
    c1,c2,c3=st.columns(3)
    latency=c1.selectbox("Required latency",["Hours or days","Minutes","Sub-second"])
    volume=c2.selectbox("Prediction volume",["Low","Moderate","High / continuous"])
    schedule=c3.selectbox("When are predictions needed?",["On a schedule","On request","As events arrive"])
    if schedule=="On a schedule": recommendation="Batch inference with an orchestrated job and versioned output table"
    elif schedule=="As events arrive": recommendation="Streaming inference with an event broker, scalable consumer, and dead-letter handling"
    elif latency=="Sub-second": recommendation="Online API serving with a warm model process, strict schema, caching, and autoscaling"
    else: recommendation="Request-driven API or queued asynchronous worker"
    st.success(f"Suggested starting pattern: {recommendation}.")

section_header("2. From notebook to a deployable artifact")
for item in ["Package preprocessing with the model.","Pin dependencies and record training code/data versions.","Define input and output schemas.","Serialize a tested model artifact.","Containerize when environment portability is needed.","Keep training and serving transformations consistent."]: st.markdown(f"- {item}")
section_header("3. Common serving techniques")
batch,api,streaming,edge=st.tabs(["Batch","API","Streaming","Edge"])
with batch: st.markdown("Schedule scoring through an orchestrator; write idempotent, partitioned outputs; backfill safely; track artifact and feature versions.")
with api: st.markdown("Expose inference through a service such as FastAPI; validate schemas; bound latency; use health checks, concurrency limits, and timeouts.")
with streaming: st.markdown("Consume events, preserve ordering where necessary, handle retries and duplicates, and route poison messages to a dead-letter queue.")
with edge: st.markdown("Optimize model size and runtime, plan update delivery, test hardware constraints, and define fallback behavior when connectivity is absent.")
section_header("4. Release safely")
for name,body in [("Shadow","Copy production traffic without affecting decisions."),("Blue-green","Maintain old and new stacks for fast switching."),("A/B test","Randomize eligible users when measuring causal product impact."),("Rollback","Preserve the previous model, configuration, and feature contract.")]: st.markdown(f"**{name}:** {body}")
section_header("5. Monitor the whole system")
ops,data,model=st.columns(3)
with ops.container(border=True):
    st.markdown("#### Operations"); st.markdown("- Latency and throughput"); st.markdown("- Error rate and availability"); st.markdown("- Queue depth and cost")
with data.container(border=True):
    st.markdown("#### Data"); st.markdown("- Missingness and schema"); st.markdown("- Feature distributions"); st.markdown("- Drift and freshness")
with model.container(border=True):
    st.markdown("#### Model"); st.markdown("- Performance after labels arrive"); st.markdown("- Calibration and fairness"); st.markdown("- Prediction distribution")