import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, matthews_corrcoef

def evaluate_predictions(predictions_file, output_file):

    # Load predictions
    df = pd.read_csv(predictions_file)
    
    # Extract true and predicted labels
    y_true = df["true_label"].values
    y_pred = df["predicted_label"].values
    
    # Calculate metrics
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred, average='weighted'),
        "Recall": recall_score(y_true, y_pred, average='weighted'),
        "Precision": precision_score(y_true, y_pred, average='weighted'),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }
    
    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    
    # Save results to Excel
    metrics_df.to_excel(output_file, index=False)
    
    return metrics

if __name__ == "__main__":
    for mode in ["contrastive", "binary"]:
        for model in os.listdir(f"results/{mode}"):
            if model.endswith(".csv"):
                model = model.split(".")[0]
                metrics = evaluate_predictions(f"results/{mode}/{model}.csv", f"results/{mode}/evaluation_results_{model}.xlsx")

