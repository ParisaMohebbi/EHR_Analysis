"""
Disease Cluster Analysis in Electronic Health Records: Insights into Mortality and Comorbidity Patterns
Submitted to the IISE Annual Conference & Expo 2025
Abstract ID: 6965
Authors: Parisa Vaghfi Mohebbi, Ahmad Salehiyan, Akash Deep
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Function to find optimal number of clusters using the highest Silhouette score
def find_optimal_clusters(data, max_clusters=10):
    # Silhouette scores for different numbers of clusters
    silhouette_scores = []
    
    # Calculate Silhouette score for different numbers of clusters
    for k in range(2, max_clusters + 1):  # Start from 2 clusters
        try:
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(data)
            
            # Calculate Silhouette score
            score = silhouette_score(data, clusters)
            silhouette_scores.append(score)
        except Exception as e:
            print(f"Error calculating Silhouette score for {k} clusters: {e}")
            silhouette_scores.append(-1)
    
    # Find the optimal number of clusters
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    
    # Plot Silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method for Optimal k')
    plt.vlines(optimal_clusters, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', colors='r')
    plt.text(optimal_clusters + 0.2, plt.ylim()[1]*0.9, f'Optimal Clusters: {optimal_clusters}')
    plt.show()
    
    return optimal_clusters

# Function to perform clustering and evaluation
def cluster_and_evaluate(data, n_clusters, method_name):
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    
    # Calculate scores
    sil_score = silhouette_score(data, clusters)
    ch_score = calinski_harabasz_score(data, clusters)
    
    print(f"\n{method_name} Clustering Metrics:")
    print(f"Silhouette Score: {sil_score:.4f}")
    print(f"Calinski-Harabasz Score: {ch_score:.4f}")
    
    return clusters

# Function to visualize clusters
def plot_clusters(data, clusters, title):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis')
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('First Component')
    plt.ylabel('Second Component')
    plt.show()

# Function to create cluster dictionaries
def create_cluster_dict(antecedents, clusters):
    cluster_dict = {}
    for antecedent, cluster in zip(antecedents, clusters):
        if cluster not in cluster_dict:
            cluster_dict[int(cluster)] = []
        cluster_dict[int(cluster)].append(antecedent)
    return cluster_dict

# Main analysis
def main(df):
    # Extract features for clustering (excluding kulczynski)
    features = df[['support', 'confidence', 'lift', 'leverage']].values
    
    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_scaled)
    print("\nPCA explained variance ratio:", pca.explained_variance_ratio_)
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(features_scaled)
    
    # Find optimal number of clusters for both methods
    print("\nFinding optimal number of clusters for PCA...")
    pca_optimal_clusters = find_optimal_clusters(pca_result)
    print(f"Optimal clusters for PCA: {pca_optimal_clusters}")
    
    print("\nFinding optimal number of clusters for t-SNE...")
    tsne_optimal_clusters = find_optimal_clusters(tsne_result)
    print(f"Optimal clusters for t-SNE: {tsne_optimal_clusters}")
    
    # Perform clustering and evaluation
    pca_clusters = cluster_and_evaluate(pca_result, pca_optimal_clusters, "PCA")
    tsne_clusters = cluster_and_evaluate(tsne_result, tsne_optimal_clusters, "t-SNE")
    
    # Visualize clusters
    plot_clusters(pca_result, pca_clusters, "PCA Clusters")
    plot_clusters(tsne_result, tsne_clusters, "t-SNE Clusters")
    
    # Create and save cluster dictionaries
    pca_dict = create_cluster_dict(df['antecedents'], pca_clusters)
    tsne_dict = create_cluster_dict(df['antecedents'], tsne_clusters)
    
    # Save dictionaries to JSON files
    with open('pca_clusters.json', 'w') as f:
        json.dump(pca_dict, f, indent=2)
    
    with open('tsne_clusters.json', 'w') as f:
        json.dump(tsne_dict, f, indent=2)
    
    return pca_dict, tsne_dict

df = pd.read_csv('results_association_rules.csv')  # Load your data
pca_clusters, tsne_clusters = main(df)
