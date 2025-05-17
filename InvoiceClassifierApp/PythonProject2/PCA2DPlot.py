import os
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px

# === CONFIGURATION ===
embeddings_folder = "embeddings"             # Folder with .json files
predictions_csv = "predictions.csv"          # CSV with 'Filename', 'PredictedLabel'

# === Load predictions ===
predictions = pd.read_csv(predictions_csv)
predictions["Filename"] = predictions["Filename"].apply(lambda x: os.path.splitext(x)[0])

print(f"✅ Loaded predictions: {len(predictions)} rows")

# === Load embeddings from JSON ===
embedding_data = []

for file in os.listdir(embeddings_folder):
    if file.endswith(".json"):
        path = os.path.join(embeddings_folder, file)
        with open(path, "r") as f:
            data = json.load(f)
            vector = data.get("Vector")
            if not vector or not isinstance(vector, list):
                print(f"⚠️ Skipped (missing Vector): {file}")
                continue

            filename = os.path.splitext(data.get("Filename", file))[0]
            label = data.get("Label", "unknown")
            embedding_data.append({
                "Filename": filename,
                "Label": label,
                "Embedding": vector
            })

print(f"✅ Loaded embeddings: {len(embedding_data)}")

df_embed = pd.DataFrame(embedding_data)

# === Merge with predictions ===
merged = pd.merge(df_embed, predictions, on="Filename", how="inner")
if merged.empty:
    print("❌ No matching entries between embeddings and predictions.csv")
    exit()

print(f"✅ Merged rows: {len(merged)}")

# === PCA Reduction ===
X = np.array(merged["Embedding"].tolist())
if X.ndim != 2 or X.shape[0] == 0:
    print("❌ Invalid embedding shape. Got:", X.shape)
    exit()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

merged["PC1"] = X_pca[:, 0]
merged["PC2"] = X_pca[:, 1]

# === Interactive 2D Plot ===
fig = px.scatter(
    merged,
    x="PC1",
    y="PC2",
    color="PredictedLabel",
    hover_name="Filename",
    title="2D PCA of KNN-Classified Invoice Embeddings"
)

fig.show()

# Optionally export to HTML
# fig.write_html("knn_embeddings_2d_plot.html")
