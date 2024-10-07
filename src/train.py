import json
import os

from torch.optim import lr_scheduler
from transformers import get_linear_schedule_with_warmup

import wandb
import torch
from tqdm import tqdm
from pytorch_lightning.cli import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_

from src.eval import evaluate_by_task
from src.models.mlp import MLPAligner
from src.models.transformer import TransformerEncoderAligner
from src.dataset.create_emb_dataset import SourceTargetEmbeddingDataset


def train(
        # training setup
        text_dataset_name: str,
        source_emb_model_name: str,
        target_emb_model_name: str,

        # aligner
        aligner_type: str = 'mlp',  # 'mlp', 'transformer'

        # [MLP] aligner setup
        n_hidden_layers: int = 0,

        # [Transformer] aligner setup
        num_blocks: int = 4,

        # training config:
        out_dir: str = 'out',
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        n_epochs: int = 100,
        patience: int = 30,
        # clip_value: float = 1.0, # [DISABLED]

        device: str = 'cuda',
):
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"Using CUDA device: {torch.cuda.current_device()}")
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU.")

    # Initialize wandb
    cfg = {
        "text_dataset_name": text_dataset_name,
        "source_emb_model_name": source_emb_model_name,
        "target_emb_model_name": target_emb_model_name,
        "n_hidden_layers": n_hidden_layers,
        "aligner_type": aligner_type,
        "num_blocks": num_blocks,
        "out_dir": out_dir,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "n_epochs": n_epochs,
        "patience": patience,
        # "clip_value": clip_value,
        "device": device,
    }
    wandb.init(project="align-text-encoders", config=cfg,
               name=out_dir.split("/")[-1])

    train_dataset = SourceTargetEmbeddingDataset(
        text_dataset_name=text_dataset_name,
        source_emb_model_name=source_emb_model_name,
        target_emb_model_name=target_emb_model_name,
        train=True,
    )
    val_dataset = SourceTargetEmbeddingDataset(
        text_dataset_name=text_dataset_name,
        source_emb_model_name=source_emb_model_name,
        target_emb_model_name=target_emb_model_name,
        train=False,
    )

    # [ALTERNATIVE] Split the dataset into train and validation, in case no validation
    # make loader to train and eval (by splitting dataset)
    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12)

    if aligner_type == 'mlp':
        model = MLPAligner(source_emb_dim=train_dataset.get_metadata()['source_emb_dim'],
                           target_emb_dim=train_dataset.get_metadata()['target_emb_dim'],
                           hidden_dim=train_dataset.get_metadata()['target_emb_dim'] // 2,
                           n_hidden_layers=n_hidden_layers)
    elif aligner_type == 'transformer':
        model = TransformerEncoderAligner(
            source_emb_dim=train_dataset.get_metadata()['source_emb_dim'],
            target_emb_dim=train_dataset.get_metadata()['target_emb_dim'],
            hidden_dim=train_dataset.get_metadata()['target_emb_dim'] // 2,
            num_blocks=num_blocks,

            # >> Default params:
            # dropout_p=0.1,
            # n_head=4,
            # seq_len=4,
        )
    else:
        raise ValueError(f"Unknown aligner type: {aligner_type}")

    model.to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # LR scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=3,
        num_training_steps=n_epochs
    )

    # create out_dir if not exists
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(dict(dataset_metadata=train_dataset.get_metadata(), model_kwargs=model.model_kwargs),
                  f, indent=2)

    # Early stopping setup
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    # Train loop
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}")
        for source_emb, target_emb in pbar:
            source_emb = source_emb.to(device)
            target_emb = target_emb.to(device)

            optimizer.zero_grad()
            pred_target_emb = model(source_emb)
            loss = torch.nn.functional.mse_loss(pred_target_emb, target_emb)
            loss.backward()

            # Gradient clipping
            # clip_grad_norm_(model.parameters(), clip_value)  # TODO consider

            optimizer.step()
            train_loss += loss.item()

            # Update progress bar with current loss
            pbar.set_postfix({'batch_train_loss': f'{loss.item():.8f}'})
            # Log to wandb (optional, might slow down training if logged too frequently)
            wandb.log({"batch_train_loss": loss.item()})

        avg_train_loss = train_loss / len(train_dataloader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for source_emb, target_emb in val_dataloader:
                source_emb = source_emb.to(device)
                target_emb = target_emb.to(device)

                pred_target_emb = model(source_emb)
                loss = torch.nn.functional.mse_loss(pred_target_emb, target_emb)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)

        # Step the LR scheduler
        scheduler.step()  # avg_val_loss

        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            'lr': scheduler.get_last_lr()[-1],
        })

        # print(f"Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
            torch.save(best_model_state, os.path.join(out_dir, "best_model.pt"))
            print(f"New best model saved with validation loss: {best_val_loss:.8f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs, having {epochs_without_improvement} epochs w/o improvement")
                break

    # Load best model for final evaluation
    model.load_state_dict(best_model_state)
    model.to(device)
    model.eval()
    final_val_loss = 0.0
    with torch.no_grad():
        for source_emb, target_emb in val_dataloader:
            source_emb = source_emb.to(device)
            target_emb = target_emb.to(device)

            pred_target_emb = model(source_emb)
            loss = torch.nn.functional.mse_loss(pred_target_emb, target_emb)
            final_val_loss += loss.item()

    final_avg_val_loss = final_val_loss / len(val_dataloader)
    print(f"Final validation loss: {final_avg_val_loss:.8f}")
    final_results = dict(final_val_loss=final_avg_val_loss)

    wandb.log(final_results)
    wandb.finish()

    # write results to a JSON file
    with open(os.path.join(out_dir, "eval_results.json"), "w") as f:
        json.dump(final_results, f, indent=2)


if __name__ == '__main__':
    # Vision objective: CIFAR-100
    train(
        text_dataset_name='coco_captions',
        target_emb_model_name='sentence-transformers/clip-ViT-L-14',
        source_emb_model_name='intfloat/e5-base-v2',
        aligner_type='transformer',
    )
