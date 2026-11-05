# ML Interview Lab

An interactive Streamlit study guide for learning data science and machine learning by rebuilding core ideas from first principles.

## Learning philosophy

This repository separates **algorithm code** from **reference content**:

- `src/exercises/` contains first-principles algorithm implementations.
- `app_pages/` contains explanations, interactive playgrounds, and interview answers.
- `src/ui.py` keeps page presentation and exercise-error handling consistent.

If an exercise is still unfinished, its page explains what is missing and keeps independent content available. Implemented functions feed the visualizations when Streamlit reruns.

Derive and implement core calculations first; use trusted libraries afterward only to verify results.

## Project map

| Exercise module | Connected page(s) |
| --- | --- |
| `gradient_descent.py` | Gradient Descent |
| `linear_regression.py` | Linear Regression |
| `regularization.py` | Regularization, Bias–Variance |
| `logistic_regression.py` | Logistic Regression |
| `bayesian.py` | Bayesian Updating |
| `neural_network.py` | Neural Networks |
| `evaluation.py` | Model Evaluation |
| `clustering.py` | Clustering Lab |
| `forecasting.py` | Forecasting Lab |

Method pages pair core concepts and mathematics with an interactive playground and interview preparation.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Suggested order

1. Gradient descent
2. Linear regression
3. Regularization and bias–variance
4. Logistic regression
5. Model evaluation
6. Bayesian updating
7. Neural networks

For each topic: predict slider behavior, implement required functions, test edge cases, compare with a trusted library, and answer interview questions without opening the expanders.

Future topics can include trees, boosting, PCA, clustering, hypothesis testing, time series, causal inference, calibration, and ML systems—but the current modules are intentionally the first milestone.

## Task-first navigation

The application is organized around the question a practitioner asks first: **what kind of problem is this?**

- **Regression** predicts continuous outcomes.
- **Classification** predicts labels or probabilities.
- **Clustering** discovers structure without target labels.
- **Forecasting** predicts future, time-indexed outcomes.
- **Foundations** such as optimization, generalization, regularization, and evaluation support several task types.

Each category roadmap explains recognition signals, baselines, validation choices, candidate methods, and tradeoffs. The shared method catalog in `src/catalog.py` also lists relevant methods marked **Not included** for comparison; those methods are not implemented in this project.

Interactive method pages keep controls next to their primary visualization. On narrow displays, Streamlit stacks those columns automatically.

The clustering and forecasting paths include interactive playgrounds and first-principles exercises in `src/exercises/clustering.py` and `src/exercises/forecasting.py`. Reference implementations live in `SOLUTIONS.md`.

The Deployment & Monitoring page provides a vendor-neutral guide to batch, API, streaming, and edge inference; safe release strategies; and production monitoring.
