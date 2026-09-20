# Learning Roadmap

Use each checkbox as evidence of understanding, not just code completion.

Begin by choosing a task path—Regression, Classification, Clustering, or Forecasting—then use the foundations and method exercises linked from that roadmap. The app catalog also compares relevant methods marked **Not included**, which are not implemented in this project.

## Phase 1 — Optimization and regression

- [ ] Implement the scalar objective, analytical gradient, and gradient-descent history
- [ ] Explain learning-rate failure modes and verify the minimum
- [ ] Implement simple OLS, prediction, residuals, and MSE
- [ ] Diagnose at least two residual-pattern problems
- [ ] Stretch: implement matrix OLS and compare with a trusted library

## Phase 2 — Generalization

- [ ] Implement L1 and L2 penalties and a regularized objective
- [ ] Explain coefficient scaling, correlated predictors, and sparsity
- [ ] Implement polynomial features for the Bias–Variance playground
- [ ] Estimate bias and variance through repeated sampling

## Phase 3 — Classification and evaluation

- [ ] Implement stable sigmoid, BCE, gradients, training, and prediction
- [ ] Verify logistic gradients and visualize a decision boundary
- [ ] Implement confusion counts and the five core threshold metrics
- [ ] Explain metric selection under imbalance and asymmetric costs
- [ ] Stretch: implement ROC and precision–recall curve points

## Phase 4 — Probability

- [ ] Implement the Beta–Binomial update and posterior mean
- [ ] Explain prior strength, conjugacy, and posterior prediction
- [ ] Stretch: implement and interpret a credible interval

## Phase 5 — Neural networks

- [ ] Implement activations and parameter initialization
- [ ] Implement and shape-check the forward pass
- [ ] Derive and implement backpropagation
- [ ] Implement updates and the training loop
- [ ] Train on XOR or two moons and compare against a framework

## Completion standard for every topic

- [ ] The page runs before and after implementation
- [ ] Edge cases have focused tests
- [ ] Results agree with a trusted reference within tolerance
- [ ] I can explain the math without reading the code
- [ ] I can answer every page interview question without opening the expander

Only after this foundation is solid: add new pages, peer feedback, progress tracking, and deployment.

## Phase 6 — Unsupervised learning

- [ ] Implement K-means distance, initialization, assignment, update, and training
- [ ] Explain how scaling and cluster geometry change results
- [ ] Implement DBSCAN neighborhood queries and cluster expansion
- [ ] Compare centroid-based and density-based definitions of a cluster
- [ ] Stretch: implement silhouette scoring

## Phase 7 — Forecasting

- [ ] Implement naive and seasonal-naive baselines
- [ ] Implement recursive moving-average forecasting
- [ ] Implement simple exponential smoothing
- [ ] Implement MAE and rolling-origin validation
- [ ] Explain horizon, leakage, seasonality, and forecast uncertainty

## Phase 8 — Production thinking

- [ ] Choose between batch, API, streaming, and edge inference from constraints
- [ ] Explain shadow, canary, blue-green, A/B, and rollback strategies
- [ ] Define operational, data, and model monitoring
