using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;

public partial class MLModel1
{
    // Modelo de entrada 
    public class ModelInput
    {
        [LoadColumn(0)]
        [ColumnName(@"text")]
        public string Text { get; set; }

        [LoadColumn(1)]
        [ColumnName(@"sentiment")]
        public string Sentiment { get; set; } = "";
    }

    // Modelo de salida 
    public class ModelOutput
    {
        [ColumnName(@"PredictedLabel")]
        public string PredictedLabel { get; set; }
    }

    private static string MLNetModelPath = Path.GetFullPath("MLModel1.mlnet");

    public static readonly Lazy<PredictionEngine<ModelInput, ModelOutput>> PredictEngine =
        new Lazy<PredictionEngine<ModelInput, ModelOutput>>(() => CreatePredictEngine(), true);

    private static PredictionEngine<ModelInput, ModelOutput> CreatePredictEngine()
    {
        var mlContext = new MLContext();
        ITransformer mlModel = mlContext.Model.Load(MLNetModelPath, out var _);
        return mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);
    }

    // Método para predecir solo la etiqueta
    public static string PredictLabel(string inputText)
    {
        var input = new ModelInput { Text = inputText };
        var predEngine = PredictEngine.Value;
        var result = predEngine.Predict(input);
        return result.PredictedLabel;
    }
}
