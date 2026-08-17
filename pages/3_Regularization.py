import streamlit as st
from src.ui import apply_global_style, page_header, section_header, todo_box, interview_box

st.set_page_config(page_title="Regularization | ML Interview Lab", page_icon="🧠", layout="wide")
apply_global_style()

page_header("Ridge vs Lasso", "Explore why shrinking coefficients can improve generalization.", eyebrow="REGULARIZATION")

n_features = st.slider("Number of features", 2, 30, 8)
correlation = st.slider("Feature correlation", 0.0, 0.95, 0.60, 0.05)
alpha = st.slider("Regularization strength (alpha)", 0.0, 20.0, 1.0, 0.1)

todo_box("Create correlated synthetic features, fit OLS, Ridge, and Lasso, then visualize coefficients. Use sklearn for comparison, but explain WHY Ridge and Lasso behave differently.")

# TODO:
# - Create a sparse true coefficient vector.
# - Generate correlated predictors controlled by the slider.
# - Compare train/test MSE.
# - Plot learned coefficients.
# - Add coefficient paths as alpha changes.

section_header("Concepts to demonstrate")
st.markdown("""
- Ridge uses an **L2 penalty** and tends to shrink coefficients smoothly.
- Lasso uses an **L1 penalty** and can drive coefficients exactly to zero.
- Regularization introduces bias in exchange for lower variance.
- Feature scaling matters because penalties act on coefficient magnitude.
""")
interview_box("Why can Lasso perform feature selection while Ridge usually cannot? Explain this geometrically, not just algebraically.")
