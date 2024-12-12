"""
Disease Cluster Analysis in Electronic Health Records: Insights into Mortality and Comorbidity Patterns
Submitted to the IISE Annual Conference & Expo 2025
Abstract ID: 6965
Authors: Parisa Vaghfi Mohebbi, Ahmad Salehiyan
"""

# Import packages
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import json

# Load the data
data = pd.read_csv('admissions_aggregated_with_patients_drg.csv')

# Process DRG codes for all patients
data['drg_codes'] = data['drg_codes'].str.replace(r"[\[\]']", "", regex=True)
drg_codes_split = data['drg_codes'].str.split(',\s*')

# Flatten and extract unique disease codes
all_drg_codes = list(set(code for codes in drg_codes_split for code in codes))

# Mapping of each code to an index
code_to_index = {code: i for i, code in enumerate(all_drg_codes)}

# Create a binary matrix for DRG codes
binary_matrix = np.zeros((len(drg_codes_split), len(all_drg_codes)), dtype=int)
for i, codes in enumerate(drg_codes_split):
    for code in codes:
        if code in code_to_index:
            binary_matrix[i, code_to_index[code]] = 1

# Encode gender using OneHotEncoder
gender_data = data['gender'].fillna('Unknown')  # Handle missing values
one_hot_encoder = OneHotEncoder(sparse=False, drop='if_binary')
gender_encoded = one_hot_encoder.fit_transform(gender_data.values.reshape(-1, 1))

# Include mortality status
mortality_status = data['hospital_expire_flag'].fillna(1).values.reshape(-1, 1) 

# Combine DRG binary matrix, gender, and mortality features
combined_features = np.hstack((binary_matrix, gender_encoded, mortality_status))

# Scale the combined data
scaler = StandardScaler()
scaled_combined_data = scaler.fit_transform(combined_features)

# PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_combined_data)

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_data = tsne.fit_transform(scaled_combined_data)

# Calculate the best number of clusters for PCA and t-SNE data
def calculate_best_clusters(data, max_clusters=10):
    best_k = 0
    best_silhouette = -1
    best_calinski = -1
    
    for k in range(2, max_clusters+1):
        kmeans = KMeans(n_clusters=k, n_init=10)
        cluster_labels = kmeans.fit_predict(data)
        
        silhouette = silhouette_score(data, cluster_labels)
        calinski = calinski_harabasz_score(data, cluster_labels)
        
        if silhouette > best_silhouette or (silhouette == best_silhouette and calinski > best_calinski):
            best_k = k
            best_silhouette = silhouette
            best_calinski = calinski
            
    return best_k

best_k_pca = calculate_best_clusters(pca_data)
best_k_tsne = calculate_best_clusters(tsne_data)

print(f'Best number of clusters for PCA: {best_k_pca}')
print(f'Best number of clusters for t-SNE: {best_k_tsne}')

# Clustering using the best number of clusters
kmeans_pca = KMeans(n_clusters=best_k_pca, n_init=10)
kmeans_tsne = KMeans(n_clusters=best_k_tsne, n_init=10)

pca_clusters = kmeans_pca.fit_predict(pca_data)
tsne_clusters = kmeans_tsne.fit_predict(tsne_data)

# Calculate silhouette and Calinski-Harabasz scores for PCA clusters
pca_silhouette = silhouette_score(pca_data, pca_clusters)
pca_calinski = calinski_harabasz_score(pca_data, pca_clusters)

print(f"PCA Clustering Silhouette Score: {pca_silhouette:.3f}")
print(f"PCA Clustering Calinski-Harabasz Score: {pca_calinski:.3f}")

# Calculate silhouette and Calinski-Harabasz scores for t-SNE clusters
tsne_silhouette = silhouette_score(tsne_data, tsne_clusters)
tsne_calinski = calinski_harabasz_score(tsne_data, tsne_clusters)

print(f"t-SNE Clustering Silhouette Score: {tsne_silhouette:.3f}")
print(f"t-SNE Clustering Calinski-Harabasz Score: {tsne_calinski:.3f}")

# Visualize clusters
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=pca_clusters, cmap='viridis', s=10)
plt.title('PCA Clustering')

plt.subplot(1, 2, 2)
plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=tsne_clusters, cmap='viridis', s=10)
plt.title('t-SNE Clustering')

plt.savefig('clustering_results_with_gender_mortality.png', dpi=300, bbox_inches='tight')
plt.show()

# Create dictionaries for PCA and t-SNE clusters
def create_cluster_dict(clustered_data, clusters, data, prefix=""):
    cluster_dict = {}
    clustered_data = pd.concat([data, pd.DataFrame({'cluster': clusters})], axis=1)
    
    for cluster in clustered_data['cluster'].unique():
        cluster_data = clustered_data[clustered_data['cluster'] == cluster].dropna()
        cluster_dict[f"{prefix}Cluster {cluster}"] = {
            'drg_codes': cluster_data['drg_codes'].tolist(),
            'patient_ids': cluster_data['subject_id'].tolist(),
            'mortality_status': cluster_data['hospital_expire_flag'].tolist()}
    
    return cluster_dict

# Save results
pca_clusters_dict = create_cluster_dict(data, pca_clusters, data, prefix="PCA ")
tsne_clusters_dict = create_cluster_dict(data, tsne_clusters, data, prefix="TSNE ")

with open('pca_results_with_gender_mortality.json', 'w') as f: 
    json.dump(pca_clusters_dict, f, indent=4)

with open('tsne_results_with_gender_mortality.json', 'w') as f: 
    json.dump(tsne_clusters_dict, f, indent=4)
