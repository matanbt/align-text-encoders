import typer
from src.train import train as train_func
from src.dataset.create_emb_dataset import create_dataset as create_dataset_func

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
    batch_size: int = 512,
    learning_rate: float = 1e-4,
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
        n_hidden_layers=n_hidden_layers,
        eval_on=eval_on,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        patience=patience,
        lr_patience=lr_patience,
        lr_factor=lr_factor,
    )


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


if __name__ == "__main__":
    app()
