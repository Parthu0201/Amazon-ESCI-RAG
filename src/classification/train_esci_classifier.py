import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)

# Constants mapping
LABEL_TO_ID = {
    "I": 0,
    "C": 1,
    "S": 2,
    "E": 3
}

ID_TO_LABEL = {
    0: "I",
    1: "C",
    2: "S",
    3: "E"
}


def setup_logging() -> None:
    """Configures standard output logging."""
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


def get_device() -> torch.device:
    """Detects available hardware device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_datasets(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and validates training and testing datasets.
    
    Args:
        train_path (str): Path to the training parquet file.
        test_path (str): Path to the testing parquet file.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and testing dataframes.
        
    Raises:
        FileNotFoundError: If input files do not exist.
        ValueError: If required columns are missing or dataframes are empty.
    """
    train_file = Path(train_path)
    test_file = Path(test_path)
    
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
        
    train_df = pd.read_parquet(train_file)
    test_df = pd.read_parquet(test_file)
    
    required_columns = ["query", "product_text", "esci_label"]
    for df, name in [(train_df, "Train"), (test_df, "Test")]:
        if df.empty:
            raise ValueError(f"{name} dataset is empty.")
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {name} dataset: {missing}")
            
    return train_df, test_df


def create_label_mapping() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Returns the ESCI label mapping dictionaries."""
    return LABEL_TO_ID, ID_TO_LABEL


def compute_class_weights(df: pd.DataFrame, label_col: str = "esci_label") -> torch.Tensor:
    """Computes class weights to handle dataset imbalance."""
    classes = np.array([0, 1, 2, 3])
    labels = df[label_col].map(LABEL_TO_ID).values
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float)


def prepare_text_inputs(df: pd.DataFrame) -> List[str]:
    """Formats query and product text for the model."""
    return [
        f"[QUERY] {str(q)} [PRODUCT] {str(p)}" 
        for q, p in zip(df["query"], df["product_text"])
    ]


class ESCIDataset(Dataset):
    """Custom PyTorch Dataset for ESCI Classification."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer: AutoTokenizer, max_length: int = 256):
        self.encodings = tokenizer(
            texts, 
            truncation=True, 
            padding="max_length", 
            max_length=max_length
        )
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
    """Computes classification evaluation metrics."""
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


class WeightedLossTrainer(Trainer):
    """Custom HuggingFace Trainer to support class weights."""
    
    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Overrides loss computation to use weighted CrossEntropyLoss."""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs.get("loss")
            
        return (loss, outputs) if return_outputs else loss


def save_results(
    output_dir: Path, 
    model: AutoModelForSequenceClassification, 
    tokenizer: AutoTokenizer, 
    eval_metrics: Dict[str, float],
    label_mapping: Dict[str, Any]
) -> None:
    """Saves model, tokenizer, label mapping, and metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    mapping_path = output_dir / "label_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(label_mapping, f, indent=4)
        
    metrics_path = output_dir / "evaluation_results.json"
    with open(metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=4)
        
    logger.info("Model Saved:")
    logger.info(output_dir.as_posix())


def _get_distribution_str(df: pd.DataFrame, col: str = "esci_label") -> str:
    """Helper to format label distribution strings."""
    counts = df[col].value_counts(normalize=True) * 100
    order = ["E", "S", "C", "I"]
    res = []
    for lbl in order:
        if lbl in counts:
            res.append(f"{lbl} : {counts[lbl]:.1f}%")
    return "\n".join(res)


def train_esci_classifier(
    train_path: str = "data/classification/train_sample.parquet",
    test_path: str = "data/classification/test_sample.parquet",
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    output_dir: str = "models/esci_classifier"
) -> None:
    """Main function to train and evaluate the ESCI classifier."""
    set_seed(42)
    
    logger.info("==================================")
    logger.info("ESCI CLASSIFIER TRAINING")
    logger.info("========================\n")
    
    device = get_device()
    logger.info(f"Device: {device.type.upper()}\n")
    
    try:
        train_df, test_df = load_datasets(train_path, test_path)
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        raise
        
    logger.info(f"Training Rows: {len(train_df)}")
    logger.info(f"Test Rows: {len(test_df)}\n")
    
    logger.info("Label Distribution:")
    logger.info(_get_distribution_str(train_df) + "\n")
    
    label_to_id, id_to_label = create_label_mapping()
    class_weights = compute_class_weights(train_df)
    
    logger.info("Class Weights:")
    weight_dict = {id_to_label[i]: round(float(w), 4) for i, w in enumerate(class_weights)}
    logger.info(f"{json.dumps(weight_dict, indent=2)}\n")
    
    logger.info("Initializing tokenizer and preparing datasets...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_texts = prepare_text_inputs(train_df)
    test_texts = prepare_text_inputs(test_df)
    
    train_labels = train_df["esci_label"].map(label_to_id).tolist()
    test_labels = test_df["esci_label"].map(label_to_id).tolist()
    
    train_dataset = ESCIDataset(train_texts, train_labels, tokenizer)
    test_dataset = ESCIDataset(test_texts, test_labels, tokenizer)
    
    logger.info("Loading base model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=4,
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=42,
        report_to="none"
    )
    
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights
    )
    
    logger.info("Starting training process...")
    trainer.train()
    
    logger.info("Starting evaluation...")
    pred_output = trainer.predict(test_dataset)
    metrics = pred_output.metrics
    preds = np.argmax(pred_output.predictions, axis=1)
    labels = pred_output.label_ids
    
    logger.info("==================================")
    logger.info("EVALUATION RESULTS")
    logger.info("==================\n")
    
    logger.info(f"Accuracy : {metrics['test_accuracy']:.2f}")
    logger.info(f"Precision: {metrics['test_precision']:.2f}")
    logger.info(f"Recall   : {metrics['test_recall']:.2f}")
    logger.info(f"F1 Score : {metrics['test_f1']:.2f}\n")
    
    logger.info("Classification Report:")
    target_names = [id_to_label[i] for i in range(4)]
    logger.info("\n" + classification_report(labels, preds, target_names=target_names))
    
    logger.info("Confusion Matrix:")
    logger.info("\n" + str(confusion_matrix(labels, preds)) + "\n")
    
    full_mapping = {
        "LABEL_TO_ID": label_to_id,
        "ID_TO_LABEL": id_to_label,
        "inference_instructions": "Model Prediction = 3 -> Convert to E, 2 -> S, 1 -> C, 0 -> I"
    }
    
    save_results(Path(output_dir), model, tokenizer, metrics, full_mapping)


if __name__ == "__main__":
    setup_logging()
    try:
        train_esci_classifier()
    except Exception as e:
        logger.error(f"Training script failed: {e}")
        sys.exit(1)
