from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, matthews_corrcoef
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def evaluate_predictions(predictions_file="predictions.csv", output_file="evaluation_results.xlsx"):
    """
    Evaluate model predictions using various metrics and save results to Excel.
    
    Args:
        predictions_file: Path to CSV file containing predictions
        output_file: Path to save Excel output with evaluation metrics
    """
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
    
    print(f"Evaluation results saved to {output_file}")
    
    return metrics

if __name__ == "__main__":
    # Load the dataset directly
    dataset = load_dataset("financial_phrasebank", "sentences_50agree", trust_remote_code=True)
    df = pd.DataFrame(dataset['train'])
    
    # Split to get just the test set
    _, test_texts, _, test_labels = train_test_split(
        df['sentence'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
    )
    
    print(f"Loaded test dataset with {len(test_texts)} examples")
    
    # Evaluate predictions
    metrics = evaluate_predictions()
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")



if __name__ == "__main__":

    predictions_file = "predictions.csv"
    output_file = "evaluation_results.xlsx"

    evaluate_predictions(predictions_file, output_file)
