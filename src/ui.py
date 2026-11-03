from collections.abc import Callable, Sequence
from typing import Any
import streamlit as st

CATEGORY_COLORS = {
    "regression": "#3b82f6",
    "classification": "#8b5cf6",
    "clustering": "#0f9f8f",
    "forecasting": "#d97706",
    "bayesian": "#e0528d",
    "neural": "#db5fbc",
    "foundations": "#64748b",
    "deployment": "#16a34a",
}

def apply_global_style():
    st.markdown("""
    <style>
    .stApp {
      background-image:
        linear-gradient(rgba(128,128,128,.028) 1px,transparent 1px),
        linear-gradient(90deg,rgba(128,128,128,.028) 1px,transparent 1px),
        radial-gradient(circle at 96% 2%,rgba(139,92,246,.10),transparent 27rem),
        radial-gradient(circle at 4% 18%,rgba(59,130,246,.085),transparent 25rem);
      background-size:30px 30px,30px 30px,auto,auto;
    }
    .block-container {padding-top:2.2rem;padding-bottom:4rem;max-width:1200px;}
    .ml-hero {
      padding:1.5rem 1.65rem;border-radius:20px;margin-bottom:1.4rem;
      background:linear-gradient(125deg,color-mix(in srgb,var(--accent) 17%,transparent),rgba(128,128,128,.035));
      border:1px solid color-mix(in srgb,var(--accent) 32%,rgba(128,128,128,.18));
      border-left:5px solid var(--accent);
      box-shadow:0 10px 30px rgba(15,23,42,.06);
    }
    .ml-eyebrow {letter-spacing:.14em;text-transform:uppercase;font-size:.74rem;color:var(--accent);margin-bottom:.4rem;font-weight:750;}
    .ml-hero h1 {margin:.1rem 0 .45rem;font-size:2.35rem}.ml-subtitle{font-size:1.08rem;opacity:.82;max-width:800px;}
    .ml-card {--card-accent:#64748b;border:1px solid rgba(128,128,128,.20);border-top:3px solid var(--card-accent);border-radius:15px;padding:1.05rem;margin-bottom:.7rem;background:color-mix(in srgb,var(--card-accent) 5%,transparent);transition:transform .15s ease,border-color .15s ease,box-shadow .15s ease;}
    .ml-card:hover {transform:translateY(-2px);border-color:color-mix(in srgb,var(--card-accent) 55%,rgba(128,128,128,.2));box-shadow:0 8px 24px rgba(15,23,42,.07)}
    .ml-card h4{margin:0 0 .35rem;color:var(--card-accent)}.ml-card p{margin:.15rem 0;opacity:.86}.ml-card-foot{font-size:.85rem;margin-top:.65rem;opacity:.68}
    .ml-section {margin-top:2rem;margin-bottom:.65rem}
    .ml-badge{display:inline-block;padding:.25rem .65rem;border-radius:999px;color:white;font-size:.74rem;font-weight:700;letter-spacing:.05em;}
    div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:15px;border-color:rgba(128,128,128,.20);background:rgba(255,255,255,.018)}
    div[data-testid="stMetric"]{background:rgba(128,128,128,.055);padding:.6rem;border-radius:12px;border-left:3px solid #3b82f6}
    div[data-testid="stExpander"]{border-radius:12px;border-color:rgba(128,128,128,.20)}
    section[data-testid="stSidebar"]{border-right:1px solid rgba(128,128,128,.16)}
    section[data-testid="stSidebar"] a[aria-current="page"]{border-radius:10px;background:linear-gradient(90deg,rgba(59,130,246,.14),rgba(139,92,246,.10))}
    .stButton > button,.stLinkButton > a{border-radius:10px}
    </style>""", unsafe_allow_html=True)

def _accent_for(label):
    value=label.lower()
    if "regression" in value or "regularization" in value: return CATEGORY_COLORS["regression"]
    if "classification" in value or "evaluation" in value: return CATEGORY_COLORS["classification"]
    if "cluster" in value or "unsupervised" in value: return CATEGORY_COLORS["clustering"]
    if "forecast" in value or "temporal" in value: return CATEGORY_COLORS["forecasting"]
    if "bayes" in value or "statistics" in value: return CATEGORY_COLORS["bayesian"]
    if "neural" in value or "deep learning" in value: return CATEGORY_COLORS["neural"]
    if "production" in value or "deployment" in value: return CATEGORY_COLORS["deployment"]
    return CATEGORY_COLORS["foundations"]

def page_header(title, subtitle, eyebrow="ML INTERVIEW LAB"):
    accent=_accent_for(f"{eyebrow} {title}")
    st.markdown(f'<div class="ml-hero" style="--accent:{accent}"><div class="ml-eyebrow">{eyebrow}</div><h1>{title}</h1><div class="ml-subtitle">{subtitle}</div></div>', unsafe_allow_html=True)

def task_header(category, title, subtitle, progress=""):
    page_header(title, subtitle, f"{category.upper()} · {progress}" if progress else category.upper())

def category_badge(category, text=None):
    color=CATEGORY_COLORS.get(category.lower(),"#64748b")
    st.markdown(f'<span class="ml-badge" style="background:{color}">{text or category}</span>',unsafe_allow_html=True)

def section_header(title): st.markdown(f'<h3 class="ml-section">{title}</h3>',unsafe_allow_html=True)

def math_panel(title, formula, explanation):
    with st.container(border=True):
        st.markdown(f"#### {title}"); st.latex(formula); st.caption(explanation)

def playground_layout():
    left,right=st.columns([1,2],gap="large"); return left.container(border=True),right.container(border=True)

def use_case_panel(use_when,avoid_when,alternatives):
    for column,title,items in zip(st.columns(3),("Use when","Be cautious when","Consider instead"),(use_when,avoid_when,alternatives)):
        with column.container(border=True):
            st.markdown(f"#### {title}"); st.markdown("\n".join(f"- {item}" for item in items))

def what_to_notice(text): st.caption(f"**What to notice:** {text}")
def method_catalog_table(methods):
    rows = ["| Method | Status | Good fit | Main tradeoff |", "|---|---|---|---|"]
    rows.extend(
        f"| {x['name']} | {x['status']} | {x['fit']} | {x['tradeoff']} |"
        for x in methods
    )
    st.markdown("\n".join(rows))
def concept_card(title,body,footer):
    accent=_accent_for(title)
    st.markdown(f'<div class="ml-card" style="--card-accent:{accent}"><h4>{title}</h4><p>{body}</p><div class="ml-card-foot">{footer}</div></div>',unsafe_allow_html=True)

def exercise_result(function:Callable[...,Any],*args:Any,**kwargs:Any)->tuple[Any,bool]:
    try: return function(*args,**kwargs),True
    except NotImplementedError:
        st.info(f"{function.__name__} is still an exercise. Implement it in src/exercises, then rerun the app."); return None,False
    except (TypeError,ValueError,FloatingPointError) as exc:
        st.warning(f"Your implementation ran, but needs debugging: {exc}"); return None,False

def implementation_guide(file_path:str,functions:Sequence[str],questions:Sequence[str],stretch_goals:Sequence[str]=())->None:
    st.markdown(f"Open **{file_path}** and implement:"); st.markdown("\n".join(f"- {name}" for name in functions))
    st.markdown("Afterward, you should be able to explain:"); st.markdown("\n".join(f"- {q}" for q in questions))
    if stretch_goals: st.markdown("**Optional stretch goals**"); st.markdown("\n".join(f"- {g}" for g in stretch_goals))

def interview_questions(items:Sequence[dict[str,str]])->None:
    for item in items:
        with st.expander(item["question"]):
            st.markdown(item["answer"])
            if item.get("testing"): st.caption(f"What the interviewer is testing: {item['testing']}")
