
using System.Collections;
using System.Text.Json;
using System.Xml.Linq;
using InvoiceClassifierApp.Models;

namespace InvoiceClassifierApp.Services;

public class InvoiceProcessor
{
    private readonly OpenAIEmbeddingService _embedding;
    private readonly KnnClassifier _knn;
    String trainingDataFilePath = "C:\\Users\\Senthil Arumugam\\Downloads\\InvoiceClassifierApp_MVP_CleanFinal\\InvoiceClassifierApp";
    public InvoiceProcessor(OpenAIEmbeddingService embedding, KnnClassifier knn)
    {
        _embedding = embedding;
        _knn = knn;
    }
    public async Task TrainAsync(List<InvoiceVector> data, string? embeddingOutputDirectory = null)
    {
        var trainingData = new List<(string Label, string Filename, float[] Vector)>();

        foreach (var doc in data)
        {
            if (doc == null || string.IsNullOrWhiteSpace(doc.Text))
            {
                Console.WriteLine($"Skipping invalid document: {doc?.Filename ?? "null"}");
                continue;
            }

            try
            {
                doc.Vector = await _embedding.GetOrLoadEmbeddingAsync(doc.Filename, doc.Text);
                trainingData.Add((doc.Label, doc.Filename, doc.Vector));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error embedding {doc.Filename}: {ex.Message}");
            }
        }

        // Train KNN
        _knn.Fit(trainingData);

        // Save batch embeddings file (optional)
        if (!string.IsNullOrWhiteSpace(trainingDataFilePath))
        {
            var embeddingsToSave = data.Select(d => new
            {
                d.Filename,
                d.Label,
                Vector = d.Vector
            }).ToList();

            var embeddingsFile = $"{trainingDataFilePath}.embeddings.json";
            var json = JsonSerializer.Serialize(embeddingsToSave, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(embeddingsFile, json);
            Console.WriteLine($"📦 Batch embeddings saved to: {embeddingsFile}");
        }

        // Also save individual files (optional)
        if (!string.IsNullOrWhiteSpace(embeddingOutputDirectory))
        {
            var documents = data.Select(d => (d.Filename, d.Text, d.Label));
            await _embedding.SaveEmbeddingsAsIndividualFilesAsync(documents, embeddingOutputDirectory);
        }
    }



    public async Task<string> ClassifyAsync(string filename, string text, string outputDirectory = "embeddings")
    {
        // Step 1: Generate or load embedding
        var vector = await _embedding.GetOrLoadEmbeddingAsync(filename, text);

        // Step 2: Predict label using KNN
        var predictedLabel = _knn.PredictLabel(vector);

        // Step 3: Prepare JSON structure
        var exportObject = new EmbeddingFileFormat
        {
            Filename = filename,
            Label = predictedLabel,
            Vector = vector
        };

        // Step 4: Save embedding to individual JSON file
        string safeName = filename.Replace(" ", "_").Replace("/", "_");
        Directory.CreateDirectory(outputDirectory);
        string outputPath = Path.Combine(outputDirectory, safeName + ".json");
        string jsonOutput = JsonSerializer.Serialize(exportObject, new JsonSerializerOptions { WriteIndented = true });
        await File.WriteAllTextAsync(outputPath, jsonOutput);

        Console.WriteLine($"✅ Embedding for {filename} saved to: {outputPath}");

        return predictedLabel;
    }

    public async Task<(string Label, double Score, string TopNeighbor)> ClassifyWithTopNeighborAsync(string filename, string text, string outputDirectory = "embeddings")
    {
        var vector = await _embedding.GetOrLoadEmbeddingAsync(filename, text);
        var (label, score, topNeighbor) = _knn.PredictLabelWithTopNeighbor(vector);

        var exportObject = new EmbeddingFileFormat
        {
            Filename = filename,
            Label = label,
            Vector = vector
        };

        string safeName = filename.Replace(" ", "_").Replace("/", "_");
        Directory.CreateDirectory(outputDirectory);
        string outputPath = Path.Combine(outputDirectory, safeName + ".json");
        string jsonOutput = JsonSerializer.Serialize(exportObject, new JsonSerializerOptions { WriteIndented = true });
        await File.WriteAllTextAsync(outputPath, jsonOutput);

        return (label, score, topNeighbor);
    }

}
