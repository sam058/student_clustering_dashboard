import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os


def load_dataset(csv_path="student-mat.csv", sep=';'):
    df = pd.read_csv(csv_path, sep=sep)
    if 'StudentID' not in df.columns:
        df['StudentID'] = df.index + 1
    return df

def preprocess_data(df, features, drop_cols=None):
    if drop_cols is None:
        drop_cols = []

    df_encoded = df.copy()
    
    # Encode categorical columns
    for col in df_encoded.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
    
    # Drop unwanted columns
    if drop_cols:
        df_encoded = df_encoded.drop(columns=drop_cols, errors='ignore')

    # Fill NaN/missing in features
    df_encoded[features] = df_encoded[features].fillna(df_encoded[features].mean())
    
    # Scale features
    X_scaled = StandardScaler().fit_transform(df_encoded[features])
    return df_encoded, X_scaled


def run_kmeans(X_scaled, n_clusters=3, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return labels, kmeans

def cluster_summary(df, labels):
    df['Cluster'] = labels
    summary = df.groupby("Cluster").mean(numeric_only=True)
    return df, summary

def cluster_centers_original(df_encoded, kmeans, drop_cols=['StudentID']):
    scaler = StandardScaler().fit(df_encoded.drop(columns=drop_cols))
    centers_original = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=df_encoded.drop(columns=drop_cols).columns
    )
    return centers_original

def plot_pca(X_scaled, labels, save_path="student_clusters_pca.png"):
    n_components = min(2, X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10,6))
    y_plot = X_pca[:,1] if X_pca.shape[1] > 1 else [0]*X_pca.shape[0]

    plt.scatter(X_pca[:,0], y_plot, c=labels, cmap='viridis', alpha=0.7)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2" if X_pca.shape[1] > 1 else "")
    plt.title("Student Clusters PCA")
    plt.colorbar(label="Cluster")
    plt.savefig(save_path, dpi=300)
    plt.show()

def save_clustered_groups(df, output_dir="clustered_output"):
    os.makedirs(output_dir,exist_ok = True)

    for cluster_num in sorted(df["Cluster"].unique()):
    #     df[df["Cluster"] == cluster_num].to_csv(f"group_cluster_{cluster_num}.csv", index=False)
    # df.to_csv("student_clustered_results.csv", index=False)
    # print("\nClustered results saved as 'student_clustered_results.csv'")
      df[df["Cluster"] == cluster_num].to_csv(
            os.path.join(output_dir, f"group_cluster_{cluster_num}.csv"), index=False
        )
      
    # Optionally, save the full dataset
    df.to_csv(os.path.join(output_dir, "student_clustered_results.csv"), index=False)
    print(f"\nClustered results saved in folder '{output_dir}'")


if __name__ == "__main__":
    df = load_dataset()
    print("Dataset shape:", df.shape)
    print(df.head())

    features = ['G1','G2','G3','studytime','absences']
    df_encoded, X_scaled = preprocess_data(df, features)
    labels, kmeans = run_kmeans(X_scaled, n_clusters=3)

    df, summary = cluster_summary(df, labels)
    sil_score = silhouette_score(X_scaled, labels)
    print(f"\nSilhouette Score: {sil_score:.3f}")

    centers = cluster_centers_original(df_encoded, kmeans)
    print("\nCluster Centers (approximate original values):")
    print(centers)

    print("\nCluster Summary (average values):")
    print(summary)

    print("\nNumber of students per cluster:")
    print(df["Cluster"].value_counts())

    # print("\n--- Student Groups by Cluster ---")
    # for cluster_num in sorted(df["Cluster"].unique()):
    #     group_students = df[df["Cluster"] == cluster_num]['StudentID'].tolist()
    #     print(f"\nGroup {cluster_num} ({len(group_students)} students):")
    #     print(", ".join(map(str, group_students)))

    plot_pca(X_scaled, labels)
    save_clustered_groups(df)
