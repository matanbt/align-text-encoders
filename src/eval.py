import datasets
import torch

from sentence_transformers import SentenceTransformer, util
"""
Performs basic evaluation on the models (CIFAR100).
Note: for extensive ones, we later use the `clip_benchmark` package.
"""


def eval_on_cifar_100(
        text_encoder_model_name: str = 'sentence-transformers/clip-ViT-L-14',  # defaults to CLIP's
        aligner_model: torch.nn.Module = torch.nn.Identity(),
        n_limit: int = 5000,
        batch_size: int = 128):
    device = 'cuda'

    # TODO generalize to other multimodal models
    model = SentenceTransformer('openai/clip-vit-large-patch14', device=device)
    text_model = SentenceTransformer(text_encoder_model_name, device=device)
    aligner_model.to(device)

    # Download the dataset
    from datasets import load_dataset
    ds = load_dataset("uoft-cs/cifar100")['test']
    ds = ds.select(range(n_limit))
    classes = ds.features['fine_label'].names

    # Encode classes
    cls_captions = [f"A photo of a {c}" for c in classes]
    emb_cls_captions = aligner_model(text_model.encode(cls_captions, convert_to_tensor=True, device=device))

    # For acc evaluation
    correct_predictions = 0
    total_samples = 0

    # Prepare the inputs
    for i in range(0, len(ds), batch_size):
        batch_image, batch_classes = ds[i:i + batch_size]['img'], torch.tensor(ds[i:i + batch_size]['fine_label'])
        batch_emb_images = model.encode(batch_image, convert_to_tensor=True, device=device, batch_size=batch_size)

        cos_scores = util.cos_sim(batch_emb_images, emb_cls_captions)
        pred_labels = torch.argmax(cos_scores, dim=1)

        # aggregate for accuracy calculation
        correct_predictions += (pred_labels.cpu() == batch_classes.cpu()).sum().item()
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


def eval_on_sts():
    pass


def eval_on_text_inversion(
        text_encoder_model_name: str = "sentence-transformers/gtr-t5-base",
        aligner_model: torch.nn.Module = torch.nn.Identity(),
        n_limit: int = 100,
        batch_size: int = 128
):
    """attempts to invert the embedding using GTR-T5-base trainer inverter, from Vec2Text"""
    import vec2text
    import torch

    # Load the embedding model (to invert)
    from src.dataset.create_emb_dataset import get_text_enc_function
    embed_func = get_text_enc_function(text_encoder_model_name)

    # Load the inversion model
    corrector = vec2text.load_pretrained_corrector("gtr-base")

    # Embed text to invert
    ds = datasets.load_dataset("jxm/nq_corpus_dpr")
    text_lst = ds['dev']['text'][:n_limit]
    text_inv_pairs = []

    # TODO add dataset slightly ood (still - short text passages)

    for batch_text in torch.utils.data.DataLoader(text_lst, batch_size=batch_size):
        embeddings = embed_func(batch_text)
        embeddings = aligner_model(embeddings)
        inv_text = vec2text.invert_embeddings(
            embeddings=embeddings.cuda(),
            corrector=corrector,
            num_steps=20,
        )
        for text, inv_text in zip(batch_text, inv_text):
            print(f"Original: {text} --> Inverted: {inv_text}")
            text_inv_pairs.append({"text": text, "inv_text": inv_text})

    # TODO add quantitative metrics: https://github.com/jxmorris12/vec2text/blob/f7e3219284a0c00bf3c0783be9aed44b521157d8/vec2text/trainers/base.py#L368

    return text_inv_pairs

@torch.no_grad()
def evaluate_by_task(
        task_name: str,
        target_emb_model_name: str,
        source_emb_model_name: str,
        aligner_model: torch.nn.Module,
        batch_size: int = 1024,
):
    if task_name == 'cifar100':
        acc_on_clip = eval_on_cifar_100(
            text_encoder_model_name=target_emb_model_name,
            batch_size=batch_size
        )
        acc_on_source = eval_on_cifar_100(
            text_encoder_model_name=source_emb_model_name,
            batch_size=batch_size
        )
        acc_on_source_w_aligned = eval_on_cifar_100(
            text_encoder_model_name=source_emb_model_name,
            aligner_model=aligner_model,
            batch_size=batch_size
        )
        results = dict(
            acc_on_clip=acc_on_clip,
            acc_on_source=acc_on_source,
            acc_on_source_w_aligned=acc_on_source_w_aligned
        )
    elif task_name == 'text_inversion':
        pairs_of_target = eval_on_text_inversion(
            text_encoder_model_name=target_emb_model_name,
            batch_size=batch_size
        )
        pairs_of_source = eval_on_text_inversion(
            text_encoder_model_name=source_emb_model_name,
            batch_size=batch_size
        )
        pairs_of_source_w_aligned = eval_on_text_inversion(
            text_encoder_model_name=source_emb_model_name,
            aligner_model=aligner_model,
            batch_size=batch_size
        )
        results = dict(
            pairs_of_target=pairs_of_target,
            pairs_of_source=pairs_of_source,
            pairs_of_source_w_aligned=pairs_of_source_w_aligned
        )
    else:
        raise ValueError(f"Unknown evaluation task: {task_name}")

    return results


if __name__ == '__main__':
    eval_on_cifar_100()
