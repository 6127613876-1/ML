import pandas as pd
from sklearn.cluster import KMeans

# Load the dataset
data_path = "C:/Users/gokul/Downloads/DataT3RR.csv"
data = pd.read_csv(data_path)

# Ensure only numerical data for clustering
X = data.select_dtypes(include=[float, int])

# K-means clustering
kmeans = KMeans(n_clusters=4, init='random', max_iter=300, random_state=0)
kmeans.fit(X)

# Get results
initial_centers = kmeans.cluster_centers_   # Initial cluster centers
final_clusters_kmeans = kmeans.labels_      # Final cluster assignments
epoch_size_kmeans = kmeans.n_iter_          # Number of iterations
error_rate_kmeans = kmeans.inertia_         # Final inertia

print("K-means Clustering:")
print("\nInitial Cluster Centers:\n", initial_centers)
print("\nFinal Clusters Labels:\n", final_clusters_kmeans)
print("\nEpoch Size:", epoch_size_kmeans)
print("\nError Rate (Inertia):", error_rate_kmeans)
