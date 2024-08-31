import json
import os

import torch
import typer

from src.models.mlp import MLP
from src.train import train as train_func
from src.dataset.create_emb_dataset import create_dataset as create_dataset_func
from src.eval import evaluate_by_task

app = typer.Typer()


@app.command()
def train(
    # training setup
    text_dataset_name: str,
    source_emb_model_name: str,
    target_emb_model_name: str,

    # aligner setup
    n_hidden_layers,

    # eval setup:
    eval_on,

    # training config:
    out_dir: str = 'out',
    batch_size: int = 512,
    learning_rate: float = 1e-2,
    n_epochs: int = 100,
    patience: int = 3,
    # clip_value: float = 1.0,
    lr_patience: int = 2,
    lr_factor: float = 0.1,
):
    train_func(
        text_dataset_name=text_dataset_name,
        source_emb_model_name=source_emb_model_name,
        target_emb_model_name=target_emb_model_name,
        n_hidden_layers=int(n_hidden_layers),
        eval_on=eval_on,
        out_dir=out_dir,
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        n_epochs=int(n_epochs),
        patience=int(patience),
        lr_patience=int(lr_patience),
        lr_factor=float(lr_factor),
    )


@app.command()
def create_dataset(
        text_dataset_name: str,
        embedder_model_name: str,
        batch_size: int = 1024,
):
    create_dataset_func(
        text_dataset_name=text_dataset_name,
        embedder_model_name=embedder_model_name,
        batch_size=batch_size,
    )


@app.command()
def evaluate(
        task_name: str,
        model_dir: str,  # e.g., `out/clip-to-e5--mock/`
):

    # Load
    with open(os.path.join(model_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    aligner_model = MLP(**metadata['model_kwargs'])  # TODO generalize the class with `model_class_name`
    aligner_model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pt")))

    # Evaluate
    results = evaluate_by_task(
        task_name=task_name,
        target_emb_model_name=metadata['target_emb_model_name'],
        source_emb_model_name=metadata['source_emb_model_name'],
        aligner_model=aligner_model,
    )

    # Save the results
    with open(os.path.join(model_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    app()
