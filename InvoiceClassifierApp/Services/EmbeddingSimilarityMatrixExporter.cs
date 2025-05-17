using System.Text.Json;
using InvoiceClassifierApp.Services;

public class EmbeddingSimilarityMatrixExporter
{
    private readonly string trainingFolder;
    private readonly string invoiceFolder;

    public EmbeddingSimilarityMatrixExporter(string trainingFolder, string invoiceFolder)
    {
        this.trainingFolder = trainingFolder;
        this.invoiceFolder = invoiceFolder;
    }

    public void ExportMatrix(string outputCsvPath)
    {
        var trainingEmbeddings = LoadEmbeddings(trainingFolder);
        var invoiceEmbeddings = LoadEmbeddings(invoiceFolder);

        var trainingKeys = trainingEmbeddings.Keys.OrderBy(k => k).ToList();
        var invoiceKeys = invoiceEmbeddings.Keys.OrderBy(k => k).ToList();

        using var writer = new StreamWriter(outputCsvPath);

        // Header row
        writer.Write("Training \\ Invoice");
        foreach (var invoiceKey in invoiceKeys)
            writer.Write($",{invoiceKey}");
        writer.WriteLine();

        // Rows: training embeddings
        foreach (var trainKey in trainingKeys)
        {
            writer.Write(trainKey);
            foreach (var invoiceKey in invoiceKeys)
            {
                var sim = CosineSimilarity(trainingEmbeddings[trainKey], invoiceEmbeddings[invoiceKey]);
                writer.Write($",{sim.ToString("F4", System.Globalization.CultureInfo.InvariantCulture)}");
            }
            writer.WriteLine();
        }
    }

    private Dictionary<string, float[]> LoadEmbeddings(string folder)
    {
        var result = new Dictionary<string, float[]>();

        foreach (var file in Directory.GetFiles(folder, "*.json"))
        {
            var json = File.ReadAllText(file);
            var embeddingObject = JsonSerializer.Deserialize<EmbeddingFileFormat>(json);

            if (embeddingObject != null && embeddingObject.Vector != null)
            {
                string key = Path.GetFileNameWithoutExtension(file);
                result[key] = embeddingObject.Vector;
            }
        }

        return result;
    }


    private double CosineSimilarity(float[] a, float[] b)
    {
        double dot = 0.0, normA = 0.0, normB = 0.0;

        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        return dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
    }
}
