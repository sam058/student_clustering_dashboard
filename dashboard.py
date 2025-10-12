import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Student Clustering Dashboard", layout="wide")

# -------------------- Title --------------------
st.title("🎓 Student Clustering Dashboard")
st.markdown("""
This dashboard visualizes student performance clusters 
based on selected academic or behavioral parameters.
""")

# -------------------- Load Dataset --------------------
df = pd.read_csv("student-mat.csv", sep=';')
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# -------------------- Feature Selection --------------------
st.subheader("⚙️ Select features for clustering")
features = st.multiselect("Choose columns for clustering", df.columns.tolist())

# -------------------- Number of clusters --------------------
st.subheader("🔢 Select number of clusters")
n_clusters = st.slider("Number of clusters", min_value=2, max_value=6, value=3)

# -------------------- Run clustering --------------------
if len(features) > 0:
    st.info("You selected: " + ", ".join(features))

    # Copy df and encode categorical columns
    df_encoded = df[features].copy()
    for col in df_encoded.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])

    # Scale features
    X_scaled = StandardScaler().fit_transform(df_encoded)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df['Cluster'] = cluster_labels

    # Silhouette score
    sil_score = silhouette_score(X_scaled, cluster_labels)
    st.subheader(f"✅ Clustering Done ({n_clusters} clusters)")
    st.write(f"Silhouette Score: **{sil_score:.3f}**")

    # -------------------- Cluster counts --------------------
    st.subheader("📊 Number of students per cluster")
    st.bar_chart(df['Cluster'].value_counts())

    # -------------------- PCA for visualization --------------------
    n_components = min(2, X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    st.subheader("📈 PCA Scatter Plot")
    fig, ax = plt.subplots(figsize=(10,6))
    y_plot = X_pca[:,1] if X_pca.shape[1] > 1 else [0]*X_pca.shape[0]
    scatter = ax.scatter(X_pca[:,0], y_plot, c=cluster_labels, cmap='viridis', alpha=0.7)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2" if X_pca.shape[1] > 1 else "")
    ax.set_title("Student Clusters (PCA)")
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    st.pyplot(fig)

   # -------------------- Correlation heatmap --------------------
from sklearn.preprocessing import LabelEncoder

# Encode categorical features so all selected columns are numeric
df_encoded = df[features].copy()
for col in df_encoded.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])

st.subheader("📊 Correlation Heatmap (all selected features)")
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# safety check

numeric_features = df_encoded.select_dtypes(include='number')
if numeric_features.shape[1] > 0:
    st.subheader("📊 Correlation Heatmap (all selected features)")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(numeric_features.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
else:
    st.warning("No numeric features available for correlation heatmap.")
