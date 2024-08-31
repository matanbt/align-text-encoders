import json
import os

from torch.optim import lr_scheduler
from transformers import get_linear_schedule_with_warmup

import wandb
from tqdm import tqdm
from pytorch_lightning.cli import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_

from src.eval import evaluate_by_task
from src.models.mlp import MLP
import torch
import datasets
from src.dataset.create_emb_dataset import get_repo_id, SourceTargetEmbeddingDataset


def train(
        # training setup
        text_dataset_name: str,
        source_emb_model_name: str,
        target_emb_model_name: str,

        # aligner setup
        n_hidden_layers: int = 0,

        # eval setup:
        eval_on: str = 'cifar100',

        # training config:
        out_dir: str = 'out',
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        n_epochs: int = 100,
        patience: int = 3,
        # clip_value: float = 1.0,
        lr_patience: int = 2,
        lr_factor: float = 0.1,
):
    dataset = SourceTargetEmbeddingDataset(
        text_dataset_name=text_dataset_name,
        source_emb_model_name=source_emb_model_name,
        target_emb_model_name=target_emb_model_name,
    )
    # make loader to train and eval (by splitting dataset)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MLP(source_emb_dim=dataset.get_metadata()['source_emb_dim'],
                target_emb_dim=dataset.get_metadata()['target_emb_dim'],
                hidden_dim=dataset.get_metadata()['target_emb_dim'] // 2,
                n_hidden_layers=n_hidden_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Add LR scheduler [TODO choose scheduler]
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=lr_patience,
    #                               monitor='val_loss',
    #                               verbose=True)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=3, num_training_steps=n_epochs * len(train_dataloader)
    # )
    def get_lr(_epoch):
        warmup_iters = n_epochs // 20
        # 1) linear warmup for warmup_iters steps
        if _epoch < warmup_iters:
            return learning_rate * (_epoch+1) / warmup_iters
        return learning_rate
        # Currently fancy logic is disabled TODO attempt
        # # 2) if it > lr_decay_iters, return min learning rate
        # if _epoch > lr_decay_iters:
        #     return min_lr
        # lr_decay_iters = n_epochs // 3
        # min_lr = learning_rate // 10
        # # 3) in between, use cosine decay down to min learning rate
        # decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        # assert 0 <= decay_ratio <= 1
        # coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        # return min_lr + coeff * (learning_rate - min_lr)

    scheduler = lr_scheduler.LambdaLR(optimizer, get_lr)

    # create out_dir if not exists
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(dict(dataset_metadata=dataset.get_metadata(), model_kwargs=model.model_kwargs),
                  f, indent=2)

    # Initialize wandb
    cfg = {
        "text_dataset_name": text_dataset_name,
        "source_emb_model_name": source_emb_model_name,
        "target_emb_model_name": target_emb_model_name,
        "n_hidden_layers": n_hidden_layers,
        "out_dir": out_dir,
        "eval_on": eval_on,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "n_epochs": n_epochs,
        "patience": patience,
        # "clip_value": clip_value,
        "lr_patience": lr_patience,
        "lr_factor": lr_factor
    }
    wandb.init(project="align-text-encoders", config=cfg)

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
            optimizer.zero_grad()
            pred_target_emb = model(source_emb)
            loss = torch.nn.functional.mse_loss(pred_target_emb, target_emb)  # cosine?
            loss.backward()

            # Gradient clipping
            # clip_grad_norm_(model.parameters(), clip_value)  # TODO read and consider

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
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Load best model for final evaluation
    model.load_state_dict(best_model_state)
    model.eval()
    final_val_loss = 0.0
    with torch.no_grad():
        for source_emb, target_emb in val_dataloader:
            pred_target_emb = model(source_emb)
            loss = torch.nn.functional.mse_loss(pred_target_emb, target_emb)
            final_val_loss += loss.item()

    final_avg_val_loss = final_val_loss / len(val_dataloader)
    print(f"Final validation loss: {final_avg_val_loss:.8f}")
    final_results = dict(final_val_loss=final_avg_val_loss)

    # Evaluation of the final goal
    final_results.update(evaluate_by_task(
        task_name=eval_on,
        target_emb_model_name=target_emb_model_name,
        source_emb_model_name=source_emb_model_name,
        aligner_model=model,
        batch_size=256,  # also actively runs the encoders
    ))

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
        source_emb_model_name='sentence-transformers/all-MiniLM-L12-v2',
        eval_on='cifar100',
    )
