import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from math import pi
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def generate_additional_figures(X_scaled, labels, K_range=range(2,10)):
    # ---------------- Silhouette Score Plot ----------------
    
    silhouette_scores = []
    for k in K_range:
        kmeans_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl_tmp = kmeans_tmp.fit_predict(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, lbl_tmp))

    optimal_k = K_range[np.argmax(silhouette_scores)]

    plt.figure(figsize=(8,5))
    plt.plot(K_range, silhouette_scores, marker='o')
    plt.axvline(optimal_k, color='red', linestyle='--')
    plt.text(optimal_k + 0.1, max(silhouette_scores)-0.05, f'Optimal K={optimal_k}', color='red', fontsize=10)
    plt.title('Silhouette Score for Different K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.savefig('silhouette_score.png', dpi=300)
    plt.show()

    # ---------------- Elbow Method ----------------
    inertia_list = []
    for k in K_range:
        kmeans_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_tmp.fit(X_scaled)
        inertia_list.append(kmeans_tmp.inertia_)

    plt.figure(figsize=(8,5))
    plt.plot(K_range, inertia_list, marker='o', color='green')
    plt.axvline(optimal_k, color='red', linestyle='--')
    plt.text(optimal_k + 0.1, max(inertia_list)*0.95, f'Chosen K={optimal_k}', color='red', fontsize=10)
    plt.title('Elbow Method: Inertia vs K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.savefig('elbow_method.png', dpi=300)
    plt.show()

    # ---------------- t-SNE Cluster Plot ----------------
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(8,6))
    for cluster in np.unique(labels):
        idx = labels == cluster
        plt.scatter(X_tsne[idx,0], X_tsne[idx,1], alpha=0.7, label=f'Cluster {cluster}')
    plt.title('t-SNE Projection of Clusters')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    plt.grid(True)
    plt.savefig('tsne_clusters.png', dpi=300)
    plt.show()
    

    print("\n✅ Additional figures generated: silhouette_score.png, elbow_method.png, tsne_clusters.png")
######### plot radars 

def plot_cluster_centers_radar(df_encoded, kmeans, scaler, features):
    
    # Inverse transform cluster centers to original feature scale
    centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=features
    )

    categories = centers.columns.tolist()
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    plt.figure(figsize=(8,6))
    for i in range(len(centers)):
        values = centers.iloc[i].tolist()
        values += values[:1]
        plt.polar(angles, values, linewidth=2, label=f'Cluster {i}')

    plt.xticks(angles[:-1], categories)
    plt.title("Cluster Centers Radar Chart")
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10, title='Clusters')
    plt.savefig('cluster_centers_radar.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✅ Radar chart saved as cluster_centers_radar.png")
