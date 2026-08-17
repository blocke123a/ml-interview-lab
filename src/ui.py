import streamlit as st

def apply_global_style():
    st.markdown(
        '''
        <style>
        .block-container {padding-top:2.2rem;padding-bottom:4rem;max-width:1200px;}
        .ml-eyebrow {letter-spacing:.14em;text-transform:uppercase;font-size:.78rem;opacity:.65;margin-bottom:.35rem;}
        .ml-subtitle {font-size:1.08rem;opacity:.78;max-width:760px;margin-bottom:1.4rem;}
        .ml-card {border:1px solid rgba(128,128,128,.25);border-radius:16px;padding:1rem;margin-bottom:.8rem;}
        .ml-card h4 {margin:0 0 .35rem 0;}
        .ml-card p {margin:.15rem 0;opacity:.82;}
        .ml-card-foot {font-size:.86rem;margin-top:.6rem;opacity:.62;}
        .ml-section {margin-top:1.8rem;margin-bottom:.55rem;}
        .todo-box {border-left:4px solid #ff9f1c;background:rgba(255,159,28,.08);padding:.9rem 1rem;border-radius:8px;margin:.8rem 0 1rem;}
        .interview-box {border-left:4px solid #6c63ff;background:rgba(108,99,255,.07);padding:.9rem 1rem;border-radius:8px;margin:.8rem 0 1rem;}
        </style>
        ''',
        unsafe_allow_html=True,
    )

def page_header(title, subtitle, eyebrow="ML INTERVIEW LAB"):
    st.markdown(f'<div class="ml-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="ml-subtitle">{subtitle}</div>', unsafe_allow_html=True)

def section_header(title):
    st.markdown(f'<h3 class="ml-section">{title}</h3>', unsafe_allow_html=True)

def concept_card(title, body, footer):
    st.markdown(
        f'<div class="ml-card"><h4>{title}</h4><p>{body}</p><div class="ml-card-foot">{footer}</div></div>',
        unsafe_allow_html=True,
    )

def todo_box(text):
    st.markdown(
        f'<div class="todo-box"><strong>Your build task</strong><br>{text}</div>',
        unsafe_allow_html=True,
    )

def interview_box(question):
    st.markdown(
        f'<div class="interview-box"><strong>Interview prompt</strong><br>{question}</div>',
        unsafe_allow_html=True,
    )
