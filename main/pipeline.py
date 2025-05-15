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
)

class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        device = features.device
        batch_size = features.size(0)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        sim_matrix = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        self_mask = torch.eye(batch_size, device=device)
        logits_mask = 1 - self_mask
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        pos_count = mask.sum(1).clamp(min=1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_count

        loss = -mean_log_prob_pos
        return loss.mean()

class SupConTrainer(Trainer):
    def __init__(self, *args, supcon_weight=0.5, temperature=0.07, **kwargs):
        super().__init__(*args, **kwargs)
        self.supcon_weight = supcon_weight
        self.ce_weight = 1 - supcon_weight
        self.supcon_loss_fn = SupConLoss(temperature)
        self.ce_loss_fn = nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        logits = outputs.logits

        ce_loss = self.ce_loss_fn(logits.view(-1, model.config.num_labels), labels.view(-1))
        cls_emb = outputs.hidden_states[-1][:, 0]
        cls_emb = F.normalize(cls_emb, p=2, dim=1)
        supcon_loss = self.supcon_loss_fn(cls_emb, labels)

        loss = self.supcon_weight * supcon_loss + self.ce_weight * ce_loss
        return (loss, outputs) if return_outputs else loss


def main(task, model_name, mode="contrastive"):  
    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.config.output_hidden_states = True
    model.config.return_dict = True

    # move to GPU/MPS if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)

    # prepare datasets
    train_ds, val_ds, test_ds = prepare_datasets(task, tokenizer)

    # metrics
    def compute_metrics(eval_pred: EvalPrediction):
        logits = eval_pred.predictions
        if isinstance(logits, (tuple, list)):
            logits = logits[0] if isinstance(logits, tuple) else np.vstack(logits)
        preds = np.argmax(logits, axis=-1)
        labels = eval_pred.label_ids
        return {"accuracy": accuracy_score(labels, preds)}

    # training args
    output_dir = f"/Users/emrekuru/Developer/CS-554/project/checkpoints/{mode}/{model_name.replace('/', '_')}/checkpoint-654"

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=f"./logs/{mode}/{model_name.replace('/', '_')}",
        logging_steps=10,
        save_total_limit=2,
        save_strategy="epoch",
        load_best_model_at_end=True,
        no_cuda=False,
    )

    # choose trainer class
    TrainerClass = SupConTrainer if mode == "contrastive" else Trainer
    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        supcon_weight=0.5,
        temperature=0.07,
    )

    # if checkpoint exists, skip training
    if os.path.exists(output_dir):
        print(f"Loading existing checkpoint from {output_dir}, skipping training.")
        model = AutoModelForSequenceClassification.from_pretrained(output_dir)
        model.to(device)
        trainer.model = model
    else:
        print(f"Training model from scratch on {device}.")
        trainer.train()

    # predict on test set
    preds_output = trainer.predict(test_ds)
    logits = preds_output.predictions
    if isinstance(logits, (tuple, list)):
        logits = logits[0] if isinstance(logits, tuple) else np.vstack(logits)
    pred_labels = np.argmax(logits, axis=1)

    # save results
    df = pd.DataFrame({
        "sentence": test_ds["sentence"],
        "true_label": test_ds["labels"],
        "predicted_label": pred_labels
    })
    out_file = f"results/{mode}/{model_name.replace('/', '_')}.csv"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Saved predictions to {out_file}")

if __name__ == "__main__":
    
    for task in ["financial"]:
        for mode in ["contrastive", "binary"]:
            for model in ["google-bert/bert-base-uncased", "answerdotai/ModernBERT-base", "distilbert/distilbert-base-uncased"]:
                main(task, model, mode=mode)
