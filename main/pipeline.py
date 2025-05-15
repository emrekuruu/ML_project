import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from dataset import prepare_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    set_seed,
    logging as hf_logging,
)

# suppress unnecessary logs
hf_logging.set_verbosity_error()

# reproducibility
set_seed(42)

# ----------------------------
# Contrastive Loss Definition
# ----------------------------
class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        batch_size = features.size(0)
        device = features.device
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        sim_matrix = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        logits_mask = 1 - torch.eye(batch_size, device=device)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        pos_count = (mask * logits_mask).sum(1).clamp(min=1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_count

        return -mean_log_prob_pos.mean()

# ----------------------------
# SupCon Trainer for Pre-training Only
# ----------------------------
class SupConTrainer(Trainer):
    def __init__(self, *args, temperature: float = 0.07, **kwargs):
        super().__init__(*args, **kwargs)
        self.supcon_loss_fn = SupConLoss(temperature)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        cls_emb = F.normalize(outputs.hidden_states[-1][:, 0], p=2, dim=1)
        loss = self.supcon_loss_fn(cls_emb, labels)
        return (loss, outputs) if return_outputs else loss

# ----------------------------
# Main workflow
# ----------------------------
def main(task, model_name, mode="contrastive"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds, val_ds, test_ds = prepare_datasets(task, tokenizer)
    try:
        num_labels = train_ds.features["labels"].num_classes
    except:
        num_labels = len(set(train_ds["labels"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.config.output_hidden_states = True
    model.config.return_dict = True
    model.to(device)

    def compute_metrics(eval_pred: EvalPrediction):
        logits = eval_pred.predictions
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(eval_pred.label_ids, preds)}

    base_dir = f"./checkpoints/{task}/{mode}/{model_name.replace('/', '_')}"
    os.makedirs(base_dir, exist_ok=True)

    # -------------------
    # Stage 1: SupCon Pre-training
    # -------------------
    if mode == "contrastive":
        supcon_dir = os.path.join(base_dir, "supcon")
        supcon_args = TrainingArguments(
            output_dir=supcon_dir,
            num_train_epochs=5,
            per_device_train_batch_size=128,
            learning_rate=5e-5,
            save_strategy="epoch",
            save_total_limit=3,
            eval_strategy="epoch",
            logging_strategy="epoch",
        )
        supcon_trainer = SupConTrainer(
            model=model,
            args=supcon_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            temperature=0.07,
        )
        supcon_trainer.train()

    # -------------------
    # Stage 2: Cross-Entropy Fine-tuning
    # -------------------
    ce_dir = os.path.join(base_dir, "ce")
    ce_args = TrainingArguments(
        output_dir=ce_dir,
        num_train_epochs=3,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=128,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_strategy="epoch",
    )
    ce_trainer = Trainer(
        model=model,
        args=ce_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    ce_trainer.train()

    # -------------------
    # Evaluation & Save
    # -------------------
    preds = ce_trainer.predict(test_ds)
    logits = preds.predictions
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    pred_labels = np.argmax(logits, axis=1)

    os.makedirs("results", exist_ok=True)
    pd.DataFrame({
        "sentence": test_ds["sentence"],
        "true_label": test_ds["labels"],
        "predicted_label": pred_labels
    }).to_csv(f"results/{task}/{mode}/{model_name.replace('/', '_')}.csv", index=False)

    # Save final CLS embeddings
    model.eval()
    all_emb = []
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(test_ds, batch_size=128):
            inputs = tokenizer(batch["sentence"], padding=True, truncation=True, return_tensors="pt").to(device)
            outs = model(**inputs, output_hidden_states=True, return_dict=True)
            all_emb.append(outs.hidden_states[-1][:, 0].cpu().numpy())
    emb_array = np.vstack(all_emb)
    df_emb = pd.DataFrame(emb_array)
    df_emb["true_label"] = test_ds["labels"]
    df_emb["predicted_label"] = pred_labels
    df_emb.to_csv(f"results/{task}/{mode}/{model_name.replace('/', '_')}_embeddings.csv", index=False)

if __name__ == "__main__":
    for task in ["news"]:
        for mode in ["contrastive", "binary"]:
            for model in ["distilbert/distilbert-base-uncased", "google-bert/bert-base-uncased"]:
                main(task, model, mode=mode)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()