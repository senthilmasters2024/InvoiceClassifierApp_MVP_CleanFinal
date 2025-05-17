import os
import json
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

# Dynamically locate the root of the .NET output directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # where Python script is
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "bin", "Debug", "net9.0"))
PREDICTIONS_CSV = os.path.join(BASE_DIR, "output//predictions.csv")
EMBEDDINGS_FOLDER = os.path.join(BASE_DIR, "embeddings")
OUTPUT_HTML = "3D_Embedding_Visualization.html"
MATCHED_CSV_OUTPUT = "predictions.csv"
EMBEDDING_DIM_FALLBACK = 1536
EMBEDDING_PREVIEW_LIMIT = 20

# === Load CSV safely with fallback for German decimal format (e.g., 9,139 or 9.139)
df_raw = pd.read_csv(PREDICTIONS_CSV, header=0, quotechar='"', engine='python')

# If extra columns due to comma-split score
if df_raw.shape[1] >= 5:
    # Combine columns 3 and 4 to fix German number formatting (e.g., 9,139)
    df_raw["SimilarityScore"] = (
        df_raw.iloc[:, 2].astype(str).str.strip()
        + df_raw.iloc[:, 3].astype(str).str.strip()
    )
    df_raw["TopNeighbor"] = df_raw.iloc[:, 4]
    df_raw["Filename"] = df_raw.iloc[:, 0]
    df_raw["PredictedLabel"] = df_raw.iloc[:, 1]
else:
    df_raw.columns = ["Filename", "PredictedLabel", "SimilarityScore", "TopNeighbor"]

# === Fix SimilarityScore from German formatting
df_raw["SimilarityScore"] = (
    df_raw["SimilarityScore"]
    .astype(str)
    .str.replace(".", "", regex=False)
    .str.replace(",", ".", regex=False)
    .astype(float)
)

# Optional normalization
if df_raw["SimilarityScore"].max() > 1.5:
    df_raw["SimilarityScore"] = df_raw["SimilarityScore"] / 10000.0

# === Final cleanup
df = df_raw[["Filename", "PredictedLabel", "SimilarityScore", "TopNeighbor"]]
df["Filename"] = df["Filename"].str.strip()

print(f"✅ Loaded and cleaned {len(df)} predictions")


# === Map embedding filenames: remove .json, keep .pdf
available_embeddings = {
    f.replace(".json", ""): f
    for f in os.listdir(EMBEDDINGS_FOLDER)
    if f.endswith(".json")
}

# === Match embedding JSON files using .pdf filename
def find_embedding_file(filename: str) -> str | None:
    return available_embeddings.get(filename)

df["embedding_file"] = df["Filename"].apply(find_embedding_file)
df = df.dropna(subset=["embedding_file"])
print(f"🔎 Matched {len(df)} files with embeddings")

# === Load matched embeddings
embeddings = []
for fname in df["embedding_file"]:
    path = os.path.join(EMBEDDINGS_FOLDER, fname)
    with open(path, "r") as f:
        content = json.load(f)
        vector = content.get("Vector") or content.get("Embedding")
        if vector:
            embeddings.append(vector)
        else:
            embeddings.append([0.0] * EMBEDDING_DIM_FALLBACK)

if not embeddings:
    print("❌ No embeddings loaded. Exiting.")
    exit(1)

# === Reduce to 3D with PCA
pca = PCA(n_components=3)
reduced = pca.fit_transform(embeddings)
df["x"], df["y"], df["z"] = reduced[:, 0], reduced[:, 1], reduced[:, 2]

# === Preview string for hover
df["EmbeddingPreview"] = [
    ", ".join(f"{v:.3f}" for v in emb[:EMBEDDING_PREVIEW_LIMIT]) + "..."
    for emb in embeddings
]

# === Plot interactive 3D chart
fig = px.scatter_3d(
    df,
    x="x", y="y", z="z",
    color="PredictedLabel",
    hover_data={
        "Filename": True,
        "SimilarityScore": True,
        "TopNeighbor": True,
        "EmbeddingPreview": True
    },
    title="📊 3D Visualization of Invoice Embeddings",
    opacity=0.85
)

fig.update_traces(marker=dict(size=6, line=dict(width=1, color='DarkSlateGrey')))
fig.update_layout(
    margin=dict(l=0, r=0, b=0, t=40),
    scene=dict(
        xaxis_title="PCA 1",
        yaxis_title="PCA 2",
        zaxis_title="PCA 3"
    )
)

# === Save output
fig.write_html(OUTPUT_HTML)
df.to_csv(MATCHED_CSV_OUTPUT, index=False)

print(f"✅ 3D plot saved to: {OUTPUT_HTML}")
print(f"📄 Matched data saved to: {MATCHED_CSV_OUTPUT}")

# === Plot interactive 2D PCA chart
fig2d = px.scatter(
    df,
    x="x", y="y",
    color="PredictedLabel",
    hover_data={
        "Filename": True,
        "SimilarityScore": True,
        "TopNeighbor": True,
        "EmbeddingPreview": True
    },
    title="📊 2D Visualization of Invoice Embeddings (PCA)"
)

fig2d.update_traces(marker=dict(size=7, line=dict(width=1, color='DarkSlateGrey')))
fig2d.update_layout(
    margin=dict(l=0, r=0, b=0, t=40),
    xaxis_title="PCA 1",
    yaxis_title="PCA 2"
)

# Save the 2D plot
OUTPUT_HTML_2D = "2D_Embedding_Visualization.html"
fig2d.write_html(OUTPUT_HTML_2D)

print(f"✅ 2D plot saved to: {OUTPUT_HTML_2D}")
