import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs, make_moons
from src.exercises.clustering import dbscan, fit_kmeans
from src.ui import exercise_result, implementation_guide, interview_questions, math_panel, playground_layout, section_header, task_header, what_to_notice

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
X,_=(make_blobs(n_samples=samples,centers=3,cluster_std=.8,random_state=seed) if shape=="Blobs" else make_moons(n_samples=samples,noise=.08,random_state=seed))
with visual:
    if algorithm=="K-means":
        result,complete=exercise_result(fit_kmeans,X,clusters,100,1e-4,int(seed))
        if complete: centroids,labels,_=result
        else:
            reference=KMeans(n_clusters=clusters,n_init=10,random_state=seed).fit(X); labels=reference.labels_; centroids=reference.cluster_centers_
        fig,ax=plt.subplots(); ax.scatter(X[:,0],X[:,1],c=labels,cmap="viridis",s=22,alpha=.8); ax.scatter(centroids[:,0],centroids[:,1],marker="X",s=180,c="white",edgecolor="black")
        what_to_notice("K-means partitions space around centroids and struggles when clusters are curved or differently sized.")
    else:
        result,complete=exercise_result(dbscan,X,epsilon,min_samples)
        labels=result if complete else DBSCAN(eps=epsilon,min_samples=min_samples).fit_predict(X)
        fig,ax=plt.subplots(); ax.scatter(X[:,0],X[:,1],c=labels,cmap="tab10",s=22,alpha=.8)
        what_to_notice("DBSCAN can recover curved groups and noise, but epsilon depends strongly on feature scale.")
    ax.set(title=f"{algorithm} on {shape.lower()}",xlabel="Feature 1",ylabel="Feature 2"); st.pyplot(fig,width="stretch")
section_header("Interview prep")
interview_questions([
 {"question":"When does K-means fail?","answer":"It struggles with non-spherical groups, unequal densities or sizes, outliers, unscaled features, and categorical variables without a meaningful Euclidean representation."},
 {"question":"How do you choose k?","answer":"Combine domain usefulness with stability, elbow plots, silhouette analysis, and validation on downstream outcomes."},
 {"question":"What does DBSCAN call noise?","answer":"Points that are neither core points nor density-reachable from a core point receive the noise label."},
 {"question":"Why scale features first?","answer":"Distance is dominated by variables with larger numerical ranges, regardless of their substantive importance."},
 {"question":"How would you evaluate clusters?","answer":"Check separation, stability, interpretability, and whether segments support a real downstream decision."}])
