from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
import pandas as pd

# Using the same dataset loaded earlier
data_path = "C:/Users/gokul/Downloads/DataT3RR.csv"
data = pd.read_csv(data_path)

# Ensure only numerical data for clustering
X = data.select_dtypes(include=[float, int])

# Hierarchical clustering
distance_matrix = pdist(X)
Z = linkage(distance_matrix, 'ward')
final_clusters_hierarchical = fcluster(Z, 4, criterion='maxclust')   # Cluster assignments
epoch_size_hierarchical = len(Z)                                     # Number of merges
error_rate_hierarchical = silhouette_score(X, final_clusters_hierarchical)  # Silhouette score

print("\nHierarchical Clustering:")
print("\nFinal Clusters Labels:\n", final_clusters_hierarchical)
print("\nEpoch Size (Number of Merges):", epoch_size_hierarchical)
print("\nSilhouette Score (Error Rate):", error_rate_hierarchical)
