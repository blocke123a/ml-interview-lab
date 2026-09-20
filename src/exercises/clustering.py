"""Unsupervised clustering exercises."""
import numpy as np

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return the straight-line distance between equally shaped vectors."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("Vectors must have equal shapes.")
    return float(np.sqrt(np.sum((a - b) ** 2)))

def initialize_centroids(X: np.ndarray, n_clusters: int, seed: int = 42) -> np.ndarray:
    """Choose distinct observations as initial K-means centroids."""
    X = np.asarray(X, float)
    if X.ndim != 2 or not 1 <= n_clusters <= len(X):
        raise ValueError("Use a 2D array and a valid cluster count.")
    rng = np.random.default_rng(seed)
    return X[rng.choice(len(X), n_clusters, replace=False)].copy()

def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each observation to its nearest centroid."""
    X, centroids = np.asarray(X, float), np.asarray(centroids, float)
    distances = np.sqrt(np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2))
    return np.argmin(distances, axis=1)

def update_centroids(X: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Return the mean feature vector for every cluster; handle empty clusters."""
    X, labels = np.asarray(X, float), np.asarray(labels)
    centroids = []
    for cluster in range(n_clusters):
        members = X[labels == cluster]
        centroids.append(members.mean(axis=0) if len(members) else X[cluster % len(X)])
    return np.asarray(centroids)

def fit_kmeans(X: np.ndarray, n_clusters: int, max_iterations: int = 100, tolerance: float = 1e-4, seed: int = 42) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Alternate assignment and update steps; return centroids, labels, and inertia history."""
    X = np.asarray(X, float)
    centroids = initialize_centroids(X, n_clusters, seed)
    inertia_history = []
    for _ in range(max_iterations):
        labels = assign_clusters(X, centroids)
        inertia_history.append(float(np.sum((X - centroids[labels]) ** 2)))
        new_centroids = update_centroids(X, labels, n_clusters)
        if np.linalg.norm(new_centroids - centroids) <= tolerance:
            centroids = new_centroids
            labels = assign_clusters(X, centroids)
            break
        centroids = new_centroids
    return centroids, labels, inertia_history

def region_query(X: np.ndarray, point_index: int, epsilon: float) -> np.ndarray:
    """Return indices within epsilon of one observation, including itself."""
    X = np.asarray(X, float)
    distances = np.sqrt(np.sum((X - X[point_index]) ** 2, axis=1))
    return np.flatnonzero(distances <= epsilon)

def dbscan(X: np.ndarray, epsilon: float, min_samples: int) -> np.ndarray:
    """Label density-connected clusters; use -1 for noise."""
    X = np.asarray(X, float)
    if epsilon <= 0 or min_samples <= 0:
        raise ValueError("epsilon and min_samples must be positive.")
    unvisited, noise = -2, -1
    labels = np.full(len(X), unvisited, dtype=int)
    cluster = 0
    for point in range(len(X)):
        if labels[point] != unvisited:
            continue
        neighbors = region_query(X, point, epsilon)
        if len(neighbors) < min_samples:
            labels[point] = noise
            continue
        labels[point] = cluster
        seeds = list(neighbors[neighbors != point])
        while seeds:
            neighbor = seeds.pop()
            if labels[neighbor] == noise:
                labels[neighbor] = cluster
            if labels[neighbor] != unvisited:
                continue
            labels[neighbor] = cluster
            nearby = region_query(X, neighbor, epsilon)
            if len(nearby) >= min_samples:
                seeds.extend(i for i in nearby if labels[i] in (unvisited, noise))
        cluster += 1
    return labels

def silhouette_score_from_scratch(X: np.ndarray, labels: np.ndarray) -> float:
    """Stretch: return the mean silhouette coefficient across non-noise observations."""
    X, labels = np.asarray(X, float), np.asarray(labels)
    valid = labels != -1
    X, labels = X[valid], labels[valid]
    clusters = np.unique(labels)
    if len(clusters) < 2:
        raise ValueError("At least two non-noise clusters are required.")
    distances = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
    scores = []
    for i, cluster in enumerate(labels):
        same = np.flatnonzero(labels == cluster)
        same = same[same != i]
        a = distances[i, same].mean() if len(same) else 0.0
        b = min(distances[i, labels == other].mean() for other in clusters if other != cluster)
        scores.append(0.0 if max(a, b) == 0 else (b - a) / max(a, b))
    return float(np.mean(scores))
