import numpy as np
import streamlit as st
from src.exercises.clustering import dbscan, fit_kmeans
from src.ui import exercise_result, interview_questions, math_panel, playground_layout, section_header, task_header, what_to_notice


def make_blobs(samples, centers=3, seed=42):
    rng = np.random.default_rng(seed)
    offsets = np.array([[-2.5, -1.5], [0.0, 2.0], [2.5, -0.5]], dtype=float)
    return offsets[np.arange(samples) % centers] + rng.normal(0, .55, (samples, 2))


def make_moons(samples, noise=.08, seed=42):
    rng = np.random.default_rng(seed)
    count = (samples + 1) // 2
    angles = np.linspace(0, np.pi, count)
    first = np.column_stack([np.cos(angles), np.sin(angles)])
    second = np.column_stack([1 - np.cos(angles), 1 - np.sin(angles) - .5])
    X = np.vstack([first, second])[:samples]
    return X + rng.normal(0, noise, X.shape)


task_header("Clustering","Clustering Lab","Compare centroid-based and density-based definitions of a group.","UNSUPERVISED")
section_header("Concept and mathematics")
st.markdown("K-means looks for compact groups around centroids. DBSCAN connects dense neighborhoods and labels isolated observations as noise.")
method=st.radio("Mathematical view",["K-means","DBSCAN"],horizontal=True)
if method=="K-means":
    math_panel("Assignment and update",r"c_i=\arg\min_j\lVert x_i-\mu_j\rVert_2^2,\qquad \mu_j=\frac{1}{|C_j|}\sum_{x_i\in C_j}x_i","Assignment chooses the nearest centroid; updating replaces each centroid with its cluster mean.")
else:
    math_panel("Density neighborhood",r"N_\varepsilon(x_i)=\{x_j:\lVert x_j-x_i\rVert_2\leq\varepsilon\}","A core point has at least min_samples observations in its epsilon-neighborhood.")
section_header("Interactive playground")
controls,visual=playground_layout()
with controls:
    algorithm=st.selectbox("Algorithm",["K-means","DBSCAN"]); shape=st.selectbox("Data shape",["Blobs","Moons"])
    samples=st.slider("Samples",100,600,250,50); seed=st.number_input("Random seed",0,999,42)
    if algorithm=="K-means": clusters=st.slider("Number of clusters",2,8,3)
    else: epsilon=st.slider("Epsilon",.05,1.0,.25,.05); min_samples=st.slider("Minimum samples",2,20,5)
X=(make_blobs(samples,seed=int(seed)) if shape=="Blobs" else make_moons(samples,noise=.08,seed=int(seed)))
with visual:
    if algorithm=="K-means":
        result,complete=exercise_result(fit_kmeans,X,clusters,100,1e-4,int(seed))
        centroids,labels,_=result if complete else (None,None,None)
        chart={"x1":X[:,0],"x2":X[:,1],"cluster":labels if labels is not None else np.zeros(samples,dtype=int)}
        what_to_notice("K-means partitions space around centroids and struggles when curved groups differ in shape or density.")
    else:
        labels=exercise_result(dbscan,X,epsilon,min_samples)[0]; chart={"x1":X[:,0],"x2":X[:,1],"cluster":labels}
        what_to_notice("DBSCAN can recover curved groups and noise, but epsilon depends strongly on feature scale.")
    st.scatter_chart(chart,x="x1",y="x2",color="cluster")
section_header("Interview prep")
interview_questions([
 {"question":"When does K-means fail?","answer":"It struggles with non-spherical groups, unequal densities or sizes, outliers, unscaled features, and categorical variables without a meaningful Euclidean representation."},
 {"question":"How do you choose k?","answer":"Combine domain usefulness with stability, elbow plots, silhouette analysis, and validation on downstream outcomes."},
 {"question":"What does DBSCAN call noise?","answer":"Points that are neither core points nor density-reachable from a core point receive the noise label."},
 {"question":"Why scale features first?","answer":"Distance is dominated by variables with larger numerical ranges, regardless of their substantive importance."},
 {"question":"How would you evaluate clusters?","answer":"Check separation, stability, interpretability, and whether segments support a real downstream decision."}])
