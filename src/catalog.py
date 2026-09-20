"""Content registry for current and future learning paths."""
METHODS = {
 "regression": [
  {"name":"Linear regression","status":"In this project","fit":"Interpretable continuous targets","tradeoff":"Assumes additive linear structure"},
  {"name":"Ridge / Lasso","status":"In this project","fit":"Many or correlated predictors","tradeoff":"Adds bias to reduce variance"},
  {"name":"Decision tree","status":"Not included","fit":"Nonlinear rules and interactions","tradeoff":"Individual trees overfit easily"},
  {"name":"Random forest","status":"Not included","fit":"Robust tabular baseline","tradeoff":"Less interpretable"},
  {"name":"Gradient boosting / XGBoost","status":"Not included","fit":"High-performance tabular prediction","tradeoff":"More tuning and leakage risk"}],
 "classification": [
  {"name":"Logistic regression","status":"In this project","fit":"Interpretable probability baseline","tradeoff":"Linear log-odds boundary"},
  {"name":"Decision tree","status":"Not included","fit":"Readable nonlinear rules","tradeoff":"Unstable without constraints"},
  {"name":"Random forest","status":"Not included","fit":"General-purpose tabular classification","tradeoff":"Larger, less transparent model"},
  {"name":"Gradient boosting / XGBoost","status":"Not included","fit":"Strong structured-data performance","tradeoff":"Sensitive to tuning"},
  {"name":"Naive Bayes","status":"Not included","fit":"Sparse text or fast baseline","tradeoff":"Strong independence assumption"},
  {"name":"Neural network","status":"In this project","fit":"Complex nonlinear data","tradeoff":"Data and compute hungry"}],
 "clustering": [
  {"name":"K-means","status":"In this project","fit":"Compact centroid-based groups","tradeoff":"Needs k and scaled features"},
  {"name":"Hierarchical clustering","status":"Not included","fit":"Nested similarity structure","tradeoff":"Can be expensive"},
  {"name":"DBSCAN","status":"In this project","fit":"Irregular dense regions and noise","tradeoff":"Scale-sensitive parameters"},
  {"name":"Gaussian mixtures","status":"Not included","fit":"Soft probabilistic membership","tradeoff":"Distribution assumptions"}],
 "forecasting": [
  {"name":"Naive / seasonal naive","status":"In this project","fit":"Essential forecast baseline","tradeoff":"No learned drivers"},
  {"name":"Exponential smoothing","status":"In this project","fit":"Level, trend, and seasonality","tradeoff":"Limited external predictors"},
  {"name":"ARIMA","status":"Not included","fit":"Autocorrelated stationary structure","tradeoff":"Requires diagnostics"},
  {"name":"Feature-based regression","status":"Not included","fit":"Calendar and external drivers","tradeoff":"Feature leakage is easy"},
  {"name":"Tree boosting","status":"Not included","fit":"Nonlinear lag and driver effects","tradeoff":"Intervals need extra work"}],
}
