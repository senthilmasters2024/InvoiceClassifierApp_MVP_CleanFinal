using InvoiceClassifierApp.Services;
using InvoiceClassifierApp.Models;
using Microsoft.Extensions.Configuration;
using System.Globalization;
using System.Text;
using System;

// === Step 1: Load the OpenAI API key from environment variable
var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
if (string.IsNullOrEmpty(apiKey))
{
    throw new InvalidOperationException("The OpenAI API key is not set. Please ensure the 'OPENAI_API_KEY' environment variable is configured.");
}

// === Step 2: Initialize the embedding service, classifier, processor, and loader
var openAiService = new OpenAIEmbeddingService(apiKey);
var knn = new KnnClassifier(k: 3); // KNN classifier with k=3
var processor = new InvoiceProcessor(openAiService, knn);
var loader = new InvoiceLoader();

// === Step 3: Load labeled training data from "TrainData" directory
var trainingData = loader.LoadTrainingDataFromPdfFolders(@"C:\Users\Senthil Arumugam\Downloads\InvoiceClassifierApp_MVP_CleanFinal\InvoiceClassifierApp\TrainData");

// === Step 4: Create output folders for each known label
var knownLabels = trainingData.Select(t => t.Label).Distinct();
foreach (var label in knownLabels)
{
    var path = Path.Combine("output", label);
    Directory.CreateDirectory(path);
    Console.WriteLine($"Created folder for: {label}");
}

Console.WriteLine("Loading invoices...");
Console.WriteLine("\nClassifying and exporting predictions...");

// === Step 5: Load invoices to be classified from the "Invoices" folder
var testInvoices = loader.LoadInvoicesToClassify(@"C:\Users\Senthil Arumugam\Downloads\InvoiceClassifierApp_MVP_CleanFinal\InvoiceClassifierApp\Invoices");

// === Step 6: Train the KNN classifier on the training data
await processor.TrainAsync(trainingData);

// === Step 7: Initialize string builders for output and CSV content
var output = new StringBuilder();
var csv = new StringBuilder();
csv.AppendLine("Filename,PredictedLabel,SimilarityScore,TopNeighbor");

// === Step 8: Loop through each invoice to classify
foreach (var invoice in testInvoices)
{
    invoice.Filename = invoice.Filename.Replace(" ", "_"); // Normalize filename
    var (predicted, score, topNeighbor) = await processor.ClassifyWithTopNeighborAsync(invoice.Filename, invoice.Text);

    // Log prediction details to console
    Console.WriteLine($"[{invoice.Filename}] → {predicted} (Score: {score:F4}) | Top: {topNeighbor}");

    // Write to CSV with invariant culture formatting to avoid decimal comma issues
    csv.AppendLine($"\"{invoice.Filename}\",\"{predicted}\",{score.ToString("F4", CultureInfo.InvariantCulture)},\"{topNeighbor}\"");

    // Prepare source and target file paths
    var sourceInvoicePath = Path.Combine(@"C:\Users\Senthil Arumugam\Downloads\InvoiceClassifierApp_MVP_CleanFinal\InvoiceClassifierApp\Invoices", invoice.Filename);
    var targetFolder = Path.Combine("output", predicted);
    var targetInvoicePath = Path.Combine(targetFolder, invoice.Filename);

    // Ensure the predicted label folder exists
    Directory.CreateDirectory(targetFolder);

    // === Step 9: Copy the invoice to its predicted output folder
    if (File.Exists(sourceInvoicePath))
    {
        File.Copy(sourceInvoicePath, targetInvoicePath, overwrite: true);
        Console.WriteLine($"Copied {invoice.Filename} to {targetFolder}");
    }
    else
    {
        Console.WriteLine($"File not found at: {sourceInvoicePath}");
    }
}

// === Step 10: Save predictions CSV to output folder
Directory.CreateDirectory("output");
var csvPath = "output/predictions.csv";
var embeddingsfolderpath = @"C:\Users\Senthil Arumugam\Downloads\InvoiceClassifierApp_MVP_CleanFinal\InvoiceClassifierApp\bin\Debug\net9.0\embeddings";

// === Step 11: Analyze similarities and export similarity matrix
var analyzer = new EmbeddingSimilarityAnalyzer(embeddingsfolderpath);
analyzer.Analyze(Path.Combine(embeddingsfolderpath, "SimilarityResults.csv"));

var exporter = new EmbeddingSimilarityMatrixExporter(embeddingsfolderpath, embeddingsfolderpath);
exporter.ExportMatrix(Path.Combine(embeddingsfolderpath, "SimilarityMatrix.csv"));

// Write predictions to disk
await File.WriteAllTextAsync(csvPath, csv.ToString());
Console.WriteLine($"\nPredictions saved to: {csvPath}");

// === Step 12: Zip the classified invoices into separate zip files for each label
Console.WriteLine("Zipping classified folders...");
foreach (var label in knownLabels)
{
    var folderPath = Path.Combine("output", label);
    var zipPath = Path.Combine("output", label + ".zip");
    if (Directory.Exists(folderPath))
    {
        if (File.Exists(zipPath))
        {
            File.Delete(zipPath);
        }
        System.IO.Compression.ZipFile.CreateFromDirectory(folderPath, zipPath);
        Console.WriteLine($"Created: {zipPath}");
    }
}
