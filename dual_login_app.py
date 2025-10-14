import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA 
from student_clustering import preprocess_data, run_kmeans, cluster_summary, plot_pca
from textblob import TextBlob
from student_clustering import load_dataset

DEFAULT_CSV = "student-mat.csv"
st.title("Student Clustering Prototype")
role = st.radio("Login as:", ["Student", "Teacher"])

#  student portal

if role == "Student":
    st.subheader("Student Portal")
    st.markdown("Enter your academic and skill details to get suggested study groups.")

    with st.form("student_form"):
        name = st.text_input("Name")
        grade = st.number_input("Average Grade", min_value=0, max_value=20, value=10)
        study_time = st.number_input("Study Hours per Week", min_value=0, max_value=40, value=5)
        absences = st.number_input("Number of Absences", min_value=0, max_value=50, value=0)
        # skills = st.text_area("Skills / Interests (comma-separated)")
        article = st.text_area("Write a short article about yourself")
        submit_student = st.form_submit_button("Get Suggested Group")

    if submit_student:
        st.success(f"Hello {name}, your group suggestions are being generated...")

        # df = pd.read_csv(DEFAULT_CSV, sep=';')
        # df['StudentID'] = df.get('StudentID', pd.Series(range(1, len(df)+1)))

        df = load_dataset(DEFAULT_CSV, sep=';')
        df['StudentID'] = df.get('StudentID', pd.Series(range(1, len(df)+1)))

        # Extract structured features from article
        def extract_features_from_article(text):
            word_count = len(text.split())
            avg_word_len = sum(len(w) for w in text.split()) / (word_count or 1)
            sentiment = TextBlob(text).sentiment.polarity
            return pd.Series({
                'word_count': word_count,
                'avg_word_len': avg_word_len,
                'sentiment': sentiment
            })

        article_features = extract_features_from_article(article)

        new_student = pd.DataFrame([{
            'G1': grade,
            'G2': grade,
            'G3': grade,
            'studytime': study_time,
            'absences': absences,
            'word_count': article_features['word_count'],
            'avg_word_len': article_features['avg_word_len'],
            'sentiment': article_features['sentiment']
        }])
        df = pd.concat([df, new_student], ignore_index=True)


        # Features to use for clustering
        base_features = ['G1','G2','G3','studytime','absences']
        all_features = base_features + ['word_count', 'avg_word_len', 'sentiment']


        # Preprocess and scale
        df_encoded, X_scaled = preprocess_data(df, all_features, drop_cols=['StudentID'])

        # Run KMeans
        labels, kmeans = run_kmeans(X_scaled, n_clusters=3)
        df, summary = cluster_summary(df, labels)

        # Current student's cluster
        current_student_id = df.iloc[-1]['StudentID']
        student_cluster = df.iloc[-1]['Cluster']
        st.info(f"You belong to Cluster {student_cluster}")

        # Suggested peers (exclude current student)
        peers = df[(df['Cluster'] == student_cluster) & (df['StudentID'] != current_student_id)]
        st.subheader("Suggested Peer Group")
        # st.dataframe(peers[['StudentID','G1','G2','G3','studytime','absences']])
        st.dataframe(peers[['StudentID'] + base_features + ['word_count','avg_word_len','sentiment']])

       


else:
    st.subheader("Teacher Portal")
    st.markdown("Upload student dataset to analyze clusters and form groups.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file, sep=';')
    else:
        st.info(f"No file uploaded. Using default dataset: {DEFAULT_CSV}")
        # df = pd.read_csv(DEFAULT_CSV, sep=';')
        df = load_dataset(DEFAULT_CSV, sep=';')

    df['StudentID'] = df.get('StudentID', pd.Series(range(1, len(df)+1)))
    st.dataframe(df.head())

    # Feature selection
    features = st.multiselect(
        "Select features for clustering",
        df.columns.tolist(),
        default=['G1','G2','G3','studytime','absences']
    )

    if features:
        # Preprocess and scale
        df_encoded, X_scaled = preprocess_data(df, features, drop_cols=['StudentID'])

        # Select number of clusters
        n_clusters = st.slider("Select number of clusters", min_value=2, max_value=6, value=3)
        labels, kmeans = run_kmeans(X_scaled, n_clusters)
        df, summary = cluster_summary(df, labels)

        # Silhouette Score
        sil_score = silhouette_score(X_scaled, labels)
        st.write(f"Silhouette Score: {sil_score:.2f}")

        # Cluster Summary
        st.subheader("Cluster Summary")
        st.dataframe(summary)

        # PCA Visualization
        st.subheader("Student Clusters PCA")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        fig, ax = plt.subplots(figsize=(8,6))
        scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='viridis', alpha=0.7)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title("Student Clusters (PCA)")
        st.pyplot(fig)

        # plot_pca(X_scaled, labels)
        #  heatmap visulaization
        st.subheader("Feature Correlation Heatmap")
        fig,ax= plt.subplots(figsize=(8,6))
        sns.heatmap(df_encoded[features].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
