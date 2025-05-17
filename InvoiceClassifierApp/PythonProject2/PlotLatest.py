import os
import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px

# === Set paths ===
base_dir = r"C:\Users\Senthil Arumugam\Downloads\InvoiceClassifierApp_MVP_CleanFinal\PythonProject2"
embedding_dir = os.path.join(base_dir, "embeddings")  # directory with .json files
predictions_path = os.path.join(base_dir, "predictions.csv")
similarity_matrix_path = os.path.join(base_dir, "SimilarityMatrix.csv")
output_plot_path = os.path.join(base_dir, "KMeans_Clustering_Visualization.html")

# === Load CSVs ===
predictions_df = pd.read_csv(predictions_path)
similarity_matrix_df = pd.read_csv(similarity_matrix_path)

# === Load Embeddings ===
embedding_data = []
for root, _, files in os.walk(embedding_dir):
    for file in files:
        if file.endswith(".json") and "Similarity" not in file:
            filepath = os.path.join(root, file)
            with open(filepath, "r") as f:
                data = json.load(f)
                if isinstance(data, list) and all(isinstance(i, float) for i in data):
                    embedding_data.append({
                        "filename": file.replace(".json", ""),
                        "embedding": data
                    })

embedding_df = pd.DataFrame(embedding_data)
if embedding_df.empty:
    raise ValueError("No valid embeddings loaded from JSON files.")

# === PCA reduction ===
embeddings = np.array(embedding_df['embedding'].tolist())
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)
embedding_df['x'] = reduced[:, 0]
embedding_df['y'] = reduced[:, 1]

# === Normalize filenames for merge ===
embedding_df['normalized_filename'] = embedding_df['filename'].str.replace('_', ' ').str.replace('.pdf', '', regex=False)
predictions_df['normalized_filename'] = predictions_df['Filename'].str.replace('.pdf', '', regex=False)

# === Merge and handle column safety ===
merged_df = embedding_df.merge(predictions_df[['normalized_filename', 'PredictedLabel']], on='normalized_filename', how='left')

# === Filter only those with predictions ===
filtered_df = merged_df.dropna(subset=['PredictedLabel'])
if filtered_df.empty:
    raise ValueError("No matching predictions found after merge.")

# === KMeans clustering ===
num_clusters = len(filtered_df['PredictedLabel'].unique())
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
filtered_df['Cluster'] = kmeans.fit_predict(filtered_df[['x', 'y']])

# === Plot ===
fig = px.scatter(
    filtered_df,
    x='x',
    y='y',
    color=filtered_df['Cluster'].astype(str),
    hover_data=['filename', 'PredictedLabel'],
    title='KMeans Clustering on Document Embeddings (PCA 2D)'
)
fig.write_html(output_plot_path)
print(f"✅ Plot saved to: {output_plot_path}")
