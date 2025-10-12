# new codeeeee

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# -------------------------
# 1. Load dataset
# -------------------------
df = pd.read_csv("student-mat.csv")  # Make sure CSV is in project folder
print("Dataset shape:", df.shape)
print(df.head())

# Optional: create a unique student identifier
df['StudentID'] = df.index + 1

# -------------------------
# 2. Encode categorical variables
# -------------------------
df_encoded = df.copy()
for col in df_encoded.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])

# -------------------------
# 3. Normalize features
# -------------------------
X_scaled = StandardScaler().fit_transform(df_encoded.drop(columns=['StudentID']))

# -------------------------
# 4. Apply KMeans clustering
# -------------------------
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Assign clusters back to original df
df['Cluster'] = cluster_labels

# -------------------------
# 5. Evaluate clustering
# -------------------------
sil_score = silhouette_score(X_scaled, cluster_labels)
print(f"\nSilhouette Score for {n_clusters} clusters: {sil_score:.3f}")

# -------------------------
# 6. Cluster centers
# -------------------------
centers_original = pd.DataFrame(
    StandardScaler().fit(df_encoded.drop(columns=['StudentID'])).inverse_transform(kmeans.cluster_centers_),
    columns=df_encoded.drop(columns=['StudentID']).columns
)
print("\nCluster Centers (approximate original values):")
print(centers_original)

# -------------------------
# 7. PCA for visualization
# -------------------------
n_features = X_scaled.shape[1]
n_components = min(2, n_features)

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10,6))
y_plot = X_pca[:,1] if X_pca.shape[1] > 1 else [0]*X_pca.shape[0]

plt.scatter(X_pca[:,0], y_plot, c=cluster_labels, cmap='viridis', alpha=0.7)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2" if X_pca.shape[1] > 1 else "")
plt.title("Student Clusters PCA")
plt.colorbar(label="Cluster")
plt.savefig("student_clusters_pca.png", dpi=300)
plt.show()

# -------------------------
# 8. Cluster summaries
# -------------------------
cluster_summary = df.groupby("Cluster").mean(numeric_only=True)
print("\nCluster Summary (average values):")
print(cluster_summary)

print("\nNumber of students per cluster:")
print(df["Cluster"].value_counts())

# -------------------------
# 9. Print and save student groups
# -------------------------
print("\n--- Student Groups by Cluster ---")
for cluster_num in sorted(df["Cluster"].unique()):
    group_students = df[df["Cluster"] == cluster_num]['StudentID'].tolist()
    print(f"\nGroup {cluster_num} ({len(group_students)} students):")
    print(", ".join(map(str, group_students)))
    
    # Save each group as a separate CSV
    df[df["Cluster"] == cluster_num].to_csv(f"group_cluster_{cluster_num}.csv", index=False)

# -------------------------
# 10. Save entire clustered dataset
# -------------------------
df.to_csv("student_clustered_results.csv", index=False)
print("\nClustered results saved as 'student_clustered_results.csv'")
