#!/usr/bin/env python3
#- step1: Initialize $K$ clustering. Here, K should be a reasonably large value, larger than the expected number of clusters.
#- step2: run one time RPCL.Here, remove clustering with no elements
#- step3: reset assignment,run one time k-mean. Here, remove clustering with no elements
#- step4: If there are $u_{k}$ deleted, go to step2. If no $u_{k}$ deleted but $u_{k}$ update, go to step3. If none, return

# this algorithms sometime produce more than clustering
# sometime even just one clustering
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate synthetic dataset with three clusters
n_samples = 300
centers = 3
X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=42, cluster_std=1.0)

# Parameters
initial_K = 10  # Start with more clusters than expected
learning_rate = 5.0*initial_K/n_samples  # Learning rate for the closest centroid
rival_penalty = 0.33*learning_rate  # Penalty rate for the second-closest centroid

# Initialize K centroids randomly from the data points
centroids = X[np.random.choice(range(X.shape[0]), initial_K, replace=False)]

def RPCL(X, centroids, learning_rate, rival_penalty):
    if len(centroids)==1:
        return centroids
    assignments = []
    centroids_copy = centroids.copy()  # Create a copy of centroids
    for x in X:
        distances = np.linalg.norm(x - centroids_copy, axis=1)
        closest, second_closest = np.argsort(distances)[:2]
        # Move closest centroid towards the point
        centroids_copy[closest] += learning_rate * (x - centroids_copy[closest])
        # Move the second closest centroid slightly away from the point
        centroids_copy[second_closest] -= rival_penalty * (x - centroids_copy[second_closest])
        assignments.append(closest)
    
    # Update centroids and remove clusters with no assigned points
    unique_assignments = set(assignments)
    new_centroids = []  # Initialize new centroids list
    for closest in unique_assignments:
        new_centroids.append(centroids_copy[closest])
    
    return np.array(new_centroids)

def assign_points(X, centroids):
    assignments = []
    for x in X:
        distances = np.linalg.norm(x - centroids, axis=1)
        closest = np.argmin(distances)
        assignments.append(closest)
    return assignments

def Kmean(X, centroids):
    # Recalculate centroids as the mean of assigned points, remove empty clusters
    assignments = assign_points(X, centroids)
    new_centroids = []
    for i in range(len(centroids)):
        points = [X[j] for j, closest in enumerate(assignments) if closest == i]
        if points:
            new_centroids.append(np.mean(points, axis=0))
    return np.array(new_centroids)

def rpcl_kmeans(X, centroids, learning_rate, rival_penalty):
    # Step 2: Run one-time RPCL update
    new_centroids = RPCL(X, centroids, learning_rate, rival_penalty)

    while True:
        #step 2 loop
        if len(new_centroids) < len(centroids):
            centroids = new_centroids
            new_centroids = RPCL(X, centroids, learning_rate, rival_penalty)
            continue
        # Step 3: Perform one-time K-means update on RPCL result
        centroids = new_centroids
        new_centroids = Kmean(X, centroids)
        
        # Step 4: Check for convergence
        if len(new_centroids) == len(centroids):
            if np.allclose(new_centroids, centroids):  # Check if centroids have stabilized
                return new_centroids  # Return final centroids



# Run the RPCL-K-means clustering
final_centroids = rpcl_kmeans(X, centroids, learning_rate, rival_penalty)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c='blue', label='Data Points', alpha=0.6)
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], c='red', marker='x', s=100, label='Final Centroids')
plt.legend()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('RPCL-K-means Clustering')
plt.show()
