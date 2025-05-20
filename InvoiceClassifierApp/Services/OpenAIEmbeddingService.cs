
using System.Net.Http.Headers;
using System.Reflection.Emit;
using System.Text;
using System.Text.Json;
using Microsoft.VisualBasic;
using OpenAI.Embeddings;
using SQLitePCL;

namespace InvoiceClassifierApp.Services;

public class OpenAIEmbeddingService
{
    private readonly string _apiKey;
    private readonly HttpClient _http;

    public OpenAIEmbeddingService(string apiKey)
    {
        _apiKey = apiKey;
        _http = new HttpClient();
        _http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);
    }

    public async Task<float[]> GetEmbeddingAsync(string text)
    {
        var payload = new
        {
            input = text,
            model = "text-embedding-3-large"
        };
        const int chunkSize = 6000; // Approx. ~500 tokens
        var chunks = ChunkText(text, chunkSize);
        // Initialize the embedding client with the API key
        OpenAI.Embeddings.EmbeddingClient client = new(payload.model, Environment.GetEnvironmentVariable("OPENAI_API_KEY"));
        var embeddings = new List<float[]>();
        foreach (var chunk in chunks)
        {
            List<string> inputs = new() { chunk };
            // Generate embeddings for the input texts
            OpenAIEmbeddingCollection collection = await client.GenerateEmbeddingsAsync(inputs);
            // Convert the embeddings to float arrays
            float[] embedding1 = collection[0].ToFloats().ToArray();
            embeddings.Add(collection[0].ToFloats().ToArray());
        }

        // Average the embeddings to return a single float array
        return AverageVectors(embeddings);


    }

    private List<string> ChunkText(string text, int chunkSize)
    {
        var chunks = new List<string>();
        for (int i = 0; i < text.Length; i += chunkSize)
        {
            chunks.Add(text.Substring(i, Math.Min(chunkSize, text.Length - i)));
        }
        return chunks;
    }

    private float[] AverageVectors(List<float[]> vectors)
    {
        int length = vectors[0].Length;
        float[] average = new float[length];

        foreach (var vec in vectors)
        {
            for (int i = 0; i < length; i++)
            {
                average[i] += vec[i];
            }
        }

        for (int i = 0; i < length; i++)
        {
            average[i] /= vectors.Count;
        }

        return average;
    }

    public async Task<float[]> GetOrLoadEmbeddingAsync(string identifier, string text)
    {
        string safeName = identifier.Replace(" ", "_").Replace("/", "_");
        string path = Path.Combine("embeddings", safeName + ".json");
        Batteries.Init();
        if (File.Exists(path))
        {
            var json = await File.ReadAllTextAsync(path);
            var loaded = JsonSerializer.Deserialize<EmbeddingFileFormat>(json);
            return loaded!.Vector;
        }

        float[] embedding = await GetEmbeddingAsync(text);

        Directory.CreateDirectory("embeddings");

        var exportObject = new EmbeddingFileFormat
        {
            Filename = identifier,
            Vector = embedding
        };

        var db = new EmbeddingStorageService("embeddings.db");


        db.SaveEmbedding(exportObject);
        Console.WriteLine("Embedding saved successfully!");

        string jsonOutput = JsonSerializer.Serialize(exportObject, new JsonSerializerOptions { WriteIndented = true });
        await File.WriteAllTextAsync(path, jsonOutput);

        return embedding;
    }


    public async Task SaveEmbeddingsAsIndividualFilesAsync(IEnumerable<(string Identifier, string Text, string Label)> documents, string outputDirectory)
    {
        Directory.CreateDirectory(outputDirectory);

        foreach (var doc in documents)
        {
            string safeName = doc.Identifier.Replace(" ", "_").Replace("/", "_");
            string singlePath = Path.Combine("embeddings", safeName + ".json");
            float[] embedding;

            // Reuse or generate embedding
            if (File.Exists(singlePath))
            {
                var json = await File.ReadAllTextAsync(singlePath);
                embedding = JsonSerializer.Deserialize<float[]>(json)!;
            }
            else
            {
                embedding = await GetEmbeddingAsync(doc.Text);
                Directory.CreateDirectory("embeddings");
                await File.WriteAllTextAsync(singlePath, JsonSerializer.Serialize(embedding));
            }

            // Create the object to save
            var exportObjectexportObject = new
            {
                doc.Identifier,
                doc.Label,
                Embedding = embedding
            };

           

            // Write to individual file in output directory
            string outputPath = Path.Combine(outputDirectory, safeName + ".json");
            string jsonOutput = JsonSerializer.Serialize(exportObjectexportObject, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(outputPath, jsonOutput);
        }

        Console.WriteLine($"✅ Individual embeddings saved to: {outputDirectory}");
    }

}
