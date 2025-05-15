import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score, matthews_corrcoef
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


def evaluate_predictions(predictions_file, output_file):
    df = pd.read_csv(predictions_file)
    y_true = df["true_label"].values
    y_pred = df["predicted_label"].values

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred, average='weighted'),
        "Recall": recall_score(y_true, y_pred, average='weighted'),
        "Precision": precision_score(y_true, y_pred, average='weighted'),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

    metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    metrics_df.to_excel(output_file, index=False)
    return metrics_df


def plot_all_model_metrics(results_root="results"):
    all_data = []
    for mode in ["contrastive", "binary"]:
        mode_path = os.path.join(results_root, mode)
        for file in os.listdir(mode_path):
            if file.startswith("evaluation_results_") and file.endswith(".xlsx"):
                model = file.replace("evaluation_results_", "").replace(".xlsx", "")
                df = pd.read_excel(os.path.join(mode_path, file))
                df["Model"] = model
                df["Mode"] = mode
                all_data.append(df)
    if not all_data:
        print("No evaluation results found.")
        return

    metrics_df = pd.concat(all_data, ignore_index=True)
    pivot_df = metrics_df.pivot_table(index=["Model", "Mode"], columns="Metric", values="Value").reset_index()

    metrics = ["Accuracy", "F1 Score", "Recall", "Precision", "MCC"]
    modes = ["contrastive", "binary"]

    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for i, mode in enumerate(modes):
        ax = axs[i]
        mode_data = pivot_df[pivot_df["Mode"] == mode]
        mode_data.set_index("Model")[metrics].plot(kind="bar", ax=ax)
        ax.set_title(f"{mode.capitalize()} Training")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig("all_model_metrics.png")


def compute_avg_interclass_distance(X, y):
    class_0 = X[y == 0]
    class_1 = X[y == 1]
    distances = pairwise_distances(class_0, class_1)
    return np.mean(distances)


def plot_model_embedding_comparison(model_name, base_path="results"):
    contrastive_path = os.path.join(base_path, "contrastive", f"{model_name}_embeddings.csv")
    binary_path = os.path.join(base_path, "binary", f"{model_name}_embeddings.csv")

    contrastive_df = pd.read_csv(contrastive_path)
    binary_df = pd.read_csv(binary_path)

    X_contrastive = contrastive_df.drop(columns=["true_label", "predicted_label"]).values
    y_contrastive = contrastive_df["true_label"].values

    X_binary = binary_df.drop(columns=["true_label", "predicted_label"]).values
    y_binary = binary_df["true_label"].values

    scaler = StandardScaler()
    X_all = np.vstack([X_contrastive, X_binary])
    X_all_scaled = scaler.fit_transform(X_all)

    tsne = TSNE(n_components=2, random_state=42)
    X_all_tsne = tsne.fit_transform(X_all_scaled)

    n_contrastive = len(X_contrastive)
    X_contrastive_tsne = X_all_tsne[:n_contrastive]
    X_binary_tsne = X_all_tsne[n_contrastive:]

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    for ax, X_tsne, y, title in zip(
        axs,
        [X_contrastive_tsne, X_binary_tsne],
        [y_contrastive, y_binary],
        ["Contrastive", "Binary"]
    ):
        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="coolwarm", alpha=0.6)
        ax.set_title(f"{title} Training Embeddings")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"embedding_tsne_{model_name}.png")
    plt.close()



if __name__ == "__main__":
    # Evaluate and save metrics
    for mode in ["contrastive", "binary"]:
        for file in os.listdir(f"results/{mode}"):
            if file.endswith(".csv") and not file.endswith("_embeddings.csv"):
                model = file.split(".")[0]
                evaluate_predictions(
                    predictions_file=f"results/{mode}/{model}.csv",
                    output_file=f"results/{mode}/evaluation_results_{model}.xlsx"
                )

    # Plot metrics
    plot_all_model_metrics()

    # Plot embeddings for common models
    contrastive_models = {
        f.replace("_embeddings.csv", "") for f in os.listdir("results/contrastive") if f.endswith("_embeddings.csv")
    }
    binary_models = {
        f.replace("_embeddings.csv", "") for f in os.listdir("results/binary") if f.endswith("_embeddings.csv")
    }
    common_models = contrastive_models & binary_models

    print(common_models)
    
    for model_name in common_models:
        plot_model_embedding_comparison(model_name)
