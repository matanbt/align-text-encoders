import os
import clip
import torch
from torchvision.datasets import CIFAR100
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer


from sentence_transformers import SentenceTransformer, util


def eval_on_cifar_100(
        text_encoder_model_name: str = 'sentence-transformers/clip-ViT-L-14',  # defaults to CLIP's
        aligner_model: torch.nn.Module = torch.nn.Identity(),
        n_limit: int = 1000,
        batch_size: int = 128):
    device = 'cuda'

    model = SentenceTransformer('sentence-transformers/clip-ViT-L-14')
    text_model = SentenceTransformer(text_encoder_model_name)

    # Download the dataset
    from datasets import load_dataset
    ds = load_dataset("uoft-cs/cifar100")['test']
    ds = ds.select(range(n_limit))
    classes = ds.features['fine_label'].names

    # Encode classes
    cls_captions = [f"A photo of a {c}" for c in classes]
    emb_cls_captions = aligner_model(text_model.encode(cls_captions))

    # For acc evaluation
    correct_predictions = 0
    total_samples = 0

    # Prepare the inputs
    for i in range(0, len(ds), batch_size):
        batch_image, batch_classes = ds[i:i+batch_size]['img'], torch.tensor(ds[i:i+batch_size]['fine_label'])
        batch_emb_images = model.encode(batch_image)

        cos_scores = util.cos_sim(batch_emb_images, emb_cls_captions)
        pred_labels = torch.argmax(cos_scores, dim=1)

        # aggregate for accuracy calculation
        correct_predictions += (pred_labels == batch_classes).sum().item()
        total_samples += len(batch_classes)

        for img, pred, true in zip(batch_image, pred_labels, batch_classes):
            print(f"Predicted: {classes[pred]} | True: {classes[true]}")
            # Show image [DEBUG]
            # import matplotlib.pyplot as plt
            # plt.imshow(img)
            # plt.axis('off')
            # plt.show()

    acc = correct_predictions / total_samples
    print(f"Accuracy: {acc}")

    return acc


if __name__ == '__main__':
    eval_on_cifar_100()
