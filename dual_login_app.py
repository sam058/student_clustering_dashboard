import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from student_clustering import preprocess_data, run_kmeans, cluster_summary, plot_pca, load_dataset
from textblob import TextBlob
from student_clustering_figures import generate_additional_figures, plot_cluster_centers_radar
from math import pi

DEFAULT_CSV = "student-mat.csv"
st.set_page_config(page_title="Student Clustering Model", layout="wide")

# ------------------ STYLING ------------------
st.markdown(
    """
    <style>
    body { background: #f8fafc; color: #1f2937; font-family: 'Inter', sans-serif; }
    .main { padding: 1rem 2rem; }
    .title { font-size: 2.2rem; font-weight: 700; color: #2563EB; text-align: center; margin-top: 2rem; margin-bottom: 1rem; }
    .subtext { text-align: center; font-size: 1.1rem; color: #4b5563; margin-bottom: 2rem; }
    .stButton button { background-color: #2563eb; color: white; border-radius: 8px; padding: 0.6rem 1.2rem; border: none; transition: 0.2s; }
    .stButton button:hover { background-color: #1d4ed8; }
    .student-card { background-color: #11827; font-size:0.95rem; border: 1px solid #e5e7eb; border-radius: 12px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); padding: 1rem; margin-bottom: 1rem; transition: all 0.2s ease-in-out; }
    .cluster-msg { background-color: #dbeafe; border-left: 6px solid #2563eb; border-radius: 10px; padding: 0.8rem; margin-top: 1rem; text-align: center; font-weight: 500; color: #1e40af; }
    .alert-success { background-color: #dcfce7; color: #166534; border-radius: 10px; padding: 0.7rem; text-align: center; font-weight: 500; }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ HOMEPAGE ------------------
st.markdown("<div class='title'>🎓 Student Clustering Model</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Discover your learning peers and analyze academic patterns easily.</div>", unsafe_allow_html=True)
st.markdown("---")

# ------------------ LOGIN SELECTION ------------------
role = st.radio("Login as:", ["Student", "Teacher"], horizontal=True)

# ------------------ STUDENT PORTAL ------------------
if role == "Student":
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;color:#2563EB;'>🎯 Student Portal</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#6b7280;'>Get personalized study group recommendations and insights.</p>", unsafe_allow_html=True)
    
    with st.form("student_form"):
        name = st.text_input("👤 Name")
        grade = st.number_input("📊 Average Grade", min_value=0, max_value=20, value=10)
        study_time = st.number_input("⏰ Study Hours per Week", min_value=0, max_value=40, value=5)
        absences = st.number_input("🚫 Number of Absences", min_value=0, max_value=50, value=0)
        submit_student = st.form_submit_button("✨ Get Suggested Group")

    if submit_student:
        with st.spinner("Analyzing your details and finding matching peers..."):
            st.success(f"Hello {name}, your group suggestions are being generated...")

            # Load dataset
            df = load_dataset(DEFAULT_CSV)
            if 'StudentID' not in df.columns:
                df['StudentID'] = range(1, len(df)+1)

            # Append new student
            new_student = pd.DataFrame([{
                'G1': grade,
                'G2': grade,
                'G3': grade,
                'studytime': study_time,
                'absences': absences,
            }])
            df = pd.concat([df, new_student], ignore_index=True)

            # Preprocess, cluster, summarize
            all_features = ['G1','G2','G3','studytime','absences']
            df_encoded, X_scaled, scaler = preprocess_data(df, all_features, drop_cols=['StudentID'])
            labels, kmeans = run_kmeans(X_scaled, n_clusters=3)
            df, summary = cluster_summary(df, labels)

            # Generate additional figures
            generate_additional_figures(X_scaled, labels)

            # Radar chart
            plot_cluster_centers_radar(df_encoded, kmeans, scaler, all_features)

            # Identify student cluster
            student_cluster = df.iloc[-1]['Cluster']

            # Display top peers
            peers = df[df['Cluster'] == student_cluster].iloc[:-1]  # exclude new student
            if peers.empty:
                st.warning("No peers available in your cluster.")
            else:
                peers['avg_grade'] = (peers['G1'] + peers['G2'] + peers['G3']) / 3
                top_n = 5
                peers_sorted = peers.sort_values(by='avg_grade', ascending=False).reset_index(drop=True)
                peers_to_show = peers_sorted.head(top_n)

                st.markdown(f"<div class='cluster-msg'>You belong to <b>Cluster {student_cluster}</b> — explore your top 5 peers below!</div>", unsafe_allow_html=True)
                st.markdown("### 👥 Top Matching Peers")
                cols = st.columns(min(5, len(peers_to_show)))

                for i, (_, row) in enumerate(peers_to_show.iterrows()):
                    word_count = row.get('word_count', 'N/A')
                    avg_word_len = row.get('avg_word_len', 'N/A')
                    sentiment = row.get('sentiment', 'N/A')

                    with cols[i % 5]:
                        st.markdown(f"""
                            <div class='student-card'>
                                <b>StudentID:</b> {row['StudentID']}<br>
                                <b>Grades:</b> {row['G1']}, {row['G2']}, {row['G3']} (Avg: {row['avg_grade']:.2f})<br>
                                <b>Studytime:</b> {row['studytime']} hrs/week<br>
                                <b>Absences:</b> {row['absences']}<br>
                                <b>Article:</b> Words={word_count}, Len={avg_word_len}, Sent={sentiment}
                            </div>
                        """, unsafe_allow_html=True)

                if len(peers_sorted) > top_n:
                    show_more_key = f"show_more_peers_cluster_{student_cluster}"
                    if st.button("Show more peers", key=show_more_key):
                        more = peers_sorted.iloc[top_n: top_n + 20]
                        st.markdown("### More peers")
                        for _, row in more.iterrows():
                            st.markdown(f"- StudentID {row['StudentID']} — Avg Grade: {row['avg_grade']:.2f}, Studytime: {row['studytime']}, Absences: {row['absences']}")

# ------------------ TEACHER PORTAL ------------------
else:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;color:#2563EB;'>📚 Teacher Portal</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#6b7280;'>Analyze, cluster, and visualize student patterns easily.</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📂 Upload Student Dataset", type=["csv"])
    df = pd.read_csv(uploaded_file, sep=';') if uploaded_file else load_dataset(DEFAULT_CSV)
    if 'StudentID' not in df.columns:
        df['StudentID'] = range(1, len(df)+1)

    features = st.multiselect("Select features for clustering", df.columns.tolist(),
                              default=['G1','G2','G3','studytime','absences'])

    if features:
        df_encoded, X_scaled, scaler = preprocess_data(df, features, drop_cols=['StudentID'])
        n_clusters = st.slider("Select number of clusters", 2, 6, 3)
        labels, kmeans = run_kmeans(X_scaled, n_clusters)
        df, summary = cluster_summary(df, labels)
        sil_score = silhouette_score(X_scaled, labels)

        st.markdown(f"<div class='alert-success'>✅ Clustering completed with Silhouette Score: <b>{sil_score:.2f}</b></div>", unsafe_allow_html=True)

        tabs = st.tabs(["📊 Summary", "🌀 PCA Plot", "🔥 Heatmap", "💾 Download CSVs"])

        with tabs[0]:
            st.markdown("### Cluster Summary Table")
            st.dataframe(summary.style.background_gradient(cmap='Greens'))

        with tabs[1]:
            st.markdown("### PCA Visualization")
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            fig, ax = plt.subplots(figsize=(8,6))
            ax.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='viridis', alpha=0.7)
            ax.set_xlabel("PCA 1")
            ax.set_ylabel("PCA 2")
            ax.set_title("PCA Projection of Clusters")
            st.pyplot(fig)

        with tabs[2]:
            st.markdown("### Feature Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(df_encoded[features].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        with tabs[3]:
            st.markdown("### Download Clustered Data")
            for cluster_num in sorted(df["Cluster"].unique()):
                csv = df[df["Cluster"] == cluster_num].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Download Cluster {cluster_num}",
                    data=csv,
                    file_name=f"cluster_{cluster_num}.csv",
                    mime='text/csv'
                )
            st.download_button("Download Full Dataset",
                               data=df.to_csv(index=False).encode('utf-8'),
                               file_name="all_clusters.csv",
                               mime='text/csv')
