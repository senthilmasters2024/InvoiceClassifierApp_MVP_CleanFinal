# 📄 Invoice Classification Agent (.NET)

A powerful document classification engine written in C# and .NET 9.0 that leverages OpenAI embeddings and cosine similarity to categorize invoices by similarity and content.

---

## 🚀 Features

- ✅ Extracts and embeds invoice text using OpenAI
- ✅ K-Nearest Neighbors (KNN) based classification
- ✅ Cosine similarity for accurate matching
- ✅ Robust file name normalization
- ✅ Outputs results as CSV and classified folders
- ✅ Embedding similarity analysis and 2D matrix export
- ✅ Optionally zips classified folders for delivery

---

## 🧰 Technologies Used

| Component           | Technology             |
|--------------------|------------------------|
| Language           | C# (.NET 9.0)          |
| Embedding Service  | OpenAI API             |
| Classification     | Custom KNN in-memory   |
| PDF Parsing        | iTextSharp / PdfPig    |
| Storage            | SQLite (optional)      |
| Visualization      | CSV, Zipping output    |
| File Structure     | Modular `Services/` and `Models/` folders |

---

## 📦 Folder Structure

```
InvoiceClassifierApp/
├── Models/
│   └── InvoiceVector.cs
├── Services/
│   ├── OpenAIEmbeddingService.cs
│   ├── InvoiceLoader.cs
│   ├── InvoiceProcessor.cs
│   ├── KnnClassifier.cs
│   ├── EmbeddingStorageService.cs
│   └── EmbeddingSimilarityAnalyzer.cs
├── Program.cs
├── output/                # Generated predictions & zips
├── TrainData/             # Labeled PDFs (by folder)
└── Invoices/              # Unlabeled invoices to classify
```

---

## ⚙️ Setup Instructions

1. **Install dependencies**
   - Visual Studio 2022+ with .NET 9.0 SDK
   - Add NuGet packages:
     ```
     dotnet add package Microsoft.Data.Sqlite
     dotnet add package SQLitePCLRaw.bundle_e_sqlite3
     dotnet add package OpenAI
     ```

2. **Set OpenAI API Key**
   - Add your API key to system environment variables:
     ```
     setx OPENAI_API_KEY "sk-..."
     ```

3. **Add training and test data**
   - Drop your labeled training PDFs into subfolders of `/TrainData/`
     - Example: `/TrainData/Healthcare/`, `/TrainData/Utilities/`
   - Drop test invoices into `/Invoices/`

4. **Run the app**
   ```
   dotnet run
   ```

---

## 🧪 Output

- `output/predictions.csv` — Classification results
- `output/<label>/` — Invoices sorted into folders
- `output/<label>.zip` — Zipped results per class
- Embedding similarity matrix in `bin/.../embeddings/SimilarityMatrix.csv`

- 
![2DEmbeddingsWithDataDisplay](https://github.com/user-attachments/assets/6d7704e9-ba68-4fa6-a6c2-fef2e1c441b5)

![2DInvoiceEmbeddings](https://github.com/user-attachments/assets/69030305-c542-4aa8-83c2-4ad07a7ff71d)

![3DInvoiceEmbeddings](https://github.com/user-attachments/assets/b5c70bbc-8711-479a-acff-d68d0884ae43)

![InvoiceClassifierSample](https://github.com/user-attachments/assets/f1563ce3-d777-42c8-a11a-718423fb6795)

---

## 🧠 Notes

- Vector similarity is based on **cosine similarity**.
- KNN `k` is configurable in `KnnClassifier(k: 3)`.
- Supports `.pdf` inputs; you can extend it to images with OCR if needed.
- Uses SQLite to optionally store embeddings for caching.

---

## 🔧 Troubleshooting

- **Cosine similarity returns 0**: Check for empty or null vectors.
- **File not found**: Confirm that filenames match exactly; normalize spaces/underscores.
- **Unhandled SQLite error**: Ensure `Batteries.Init()` is called and bundle package is installed.

---

## 📌 TODOs (Future Enhancements)

- Add a UI with WinForms or WPF
- Enable retraining from feedback
- Move embeddings to a database
- Upgrade to Azure AI Search + Python for large-scale search

---

## 👨‍💻 Author
Senthil Masters — Software Engineer & AI Solution Developer  
> Building intelligent invoice agents with .NET and Azure





