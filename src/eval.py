import collections
import random
from typing import List

import datasets
import nltk
import torch
import numpy as np
import evaluate
from scipy.stats import sem

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
    raise NotImplementedError("This function is not yet updated to the new changes (e.g., OpenAI API vs SeT's)")
    device = 'cuda'

    # TODO generalize to other multimodal models
    model = SentenceTransformer('openai/clip-vit-large-patch14', device=device)  # TODO should be opened with CLIP's HF class
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
    pass  # TODO


def eval_on_text_inversion(
        text_encoder_model_name: str = "sentence-transformers/gtr-t5-base",
        aligner_model: torch.nn.Module = torch.nn.Identity(),
        n_limit: int = 2500,
        batch_size: int = 16,
        data_to_invert: str = 'nq',  # 'nq' for in-domain; 'msmarco' for slightly out-of-domain
        trained_aligner_dir: str = None,
):
    """attempts to invert the embedding using GTR-T5-base trainer inverter, from Vec2Text"""
    import vec2text
    import torch

    # Load the embedding model (to invert)
    from src.dataset.create_emb_dataset import get_text_enc_function
    embed_func = get_text_enc_function(text_encoder_model_name)
    aligner_model.cuda()

    # Load the inversion model
    # corrector = vec2text.load_pretrained_corrector("gtr-base")

    if trained_aligner_dir is not None:  # we use the aligner (or the source)
        aligner_mode = "identity" if isinstance(aligner_model, torch.nn.Identity) else "aligner"
        inversion_model = vec2text.models.InversionModel.from_pretrained(
            "jxm/gtr__nq__32",
            # should correspond to the name in `model_utils.py -> load_embedder_and_tokenizer()`
            embedder_model_name=f"CUSTOM-EMBEDDER___{trained_aligner_dir}___{aligner_mode}",#"my mock embedder name",
        )

    else:
        inversion_model = vec2text.models.InversionModel.from_pretrained("jxm/gtr__nq__32")
    model = vec2text.models.CorrectorEncoderModel.from_pretrained("jxm/gtr__nq__32__correct")
    corrector = vec2text.load_corrector(inversion_model, model)

    # Embed text to invert
    if data_to_invert == 'nq':
        ds = datasets.load_dataset("jxm/nq_corpus_dpr")
        text_lst = ds['dev']['text']
        text_lst = [t[:32 * 4] for t in text_lst]  # limit to roughly 32 tokens
    elif data_to_invert == 'msmarco':
        ds = datasets.load_dataset("Tevatron/msmarco-passage-corpus")
        text_lst = ds['train']['text']
        text_lst = [t[:32 * 4] for t in text_lst]  # limit to roughly 32 tokens
    # elif [??]: # TODO add dataset slightly OOD (still - short text passages)
    else:
        raise ValueError(f"Unknown data_to_invert: {data_to_invert}")

    # random sample of `n_limit` texts
    random.seed(42)
    text_lst = random.sample(text_lst, n_limit)

    text_inv_pairs = []

    # aggregate cosine to average:
    cos_sims = []

    for batch_text in torch.utils.data.DataLoader(text_lst, batch_size=batch_size):
        batch_embeddings = aligner_model(embed_func(batch_text).cuda())
        batch_inv_text = vec2text.invert_embeddings(
            embeddings=batch_embeddings,
            corrector=corrector,
            num_steps=100,
            sequence_beam_width=4,
        )
        batch_inv_embeddings = aligner_model(embed_func(batch_inv_text).cuda())

        # Metric I: Cosine sim.:
        batch_cos_sims = torch.cosine_similarity(batch_embeddings, batch_inv_embeddings)
        for i, (text, inv_text) in enumerate(zip(batch_text, batch_inv_text)):
            print(f"Original: {text} --> Inverted: {inv_text}")
            text_inv_pairs.append({"text": text, "inv_text": inv_text,
                                   "cosine": batch_cos_sims[i].item()})
        cos_sims.append(batch_cos_sims)

    cos_sims = torch.cat(cos_sims)
    # Metric II: Text comparison metrics
    text_comp_metrics = _calc_text_comparison_metrics(
        predictions_str=[d['inv_text'] for d in text_inv_pairs],
        references_str=[d['text'] for d in text_inv_pairs]
    )
    # TODO show text_comp_metrics per text (like cos), and then in summary (as now)
    return dict(
        text_pairs=text_inv_pairs,
        cosine_mean=cos_sims.mean().item(),
        cosine_std=cos_sims.std().item(),
        **text_comp_metrics
    )


def _calc_text_comparison_metrics(
    predictions_str: List[str],
    references_str: List[str]
) -> dict:
    """
    Calculates various measures to compare the inverted text with the original text.
    Taken from https://github.com/jxmorris12/vec2text/blob/f7e3219284a0c00bf3c0783be9aed44b521157d8/vec2text/trainers/base.py
    """
    assert len(predictions_str) == len(references_str)
    num_preds = len(predictions_str)

    ###########################################################
    # Compute token, precision, recall, and ngram-level metrics.
    precision_sum = 0.0
    recall_sum = 0.0
    num_overlapping_words = []
    num_overlapping_bigrams = []
    num_overlapping_trigrams = []
    num_true_words = []
    num_pred_words = []
    f1s = []
    for i in range(num_preds):
        true_words = nltk.tokenize.word_tokenize(references_str[i])
        pred_words = nltk.tokenize.word_tokenize(predictions_str[i])
        num_true_words.append(len(true_words))
        num_pred_words.append(len(pred_words))

        true_words_set = set(true_words)
        pred_words_set = set(pred_words)
        TP = len(true_words_set & pred_words_set)
        FP = len(true_words_set) - len(true_words_set & pred_words_set)
        FN = len(pred_words_set) - len(true_words_set & pred_words_set)

        precision = (TP) / (TP + FP + 1e-20)
        recall = (TP) / (TP + FN + 1e-20)

        try:
            f1 = (2 * precision * recall) / (precision + recall + 1e-20)
        except ZeroDivisionError:
            f1 = 0.0
        f1s.append(f1)

        precision_sum += precision
        recall_sum += recall

        ############################################################
        def count_overlapping_ngrams(s1: str, s2: str, n: int) -> int:
            ngrams_1 = nltk.ngrams(s1, n)
            ngrams_2 = nltk.ngrams(s2, n)
            ngram_counts_1 = collections.Counter(ngrams_1)
            ngram_counts_2 = collections.Counter(ngrams_2)
            total = 0
            for ngram, count in ngram_counts_1.items():
                total += min(count, ngram_counts_2[ngram])
            return total

        num_overlapping_words.append(
            count_overlapping_ngrams(true_words, pred_words, 1)
        )
        num_overlapping_bigrams.append(
            count_overlapping_ngrams(true_words, pred_words, 2)
        )
        num_overlapping_trigrams.append(
            count_overlapping_ngrams(true_words, pred_words, 3)
        )
    set_token_metrics = {
        "token_set_precision": (precision_sum / num_preds),
        "token_set_recall": (recall_sum / num_preds),
        "token_set_f1": np.array(f1s).mean(),
        "token_set_f1_sem": sem(f1s),
        "token_set_f1_sd": np.array(f1s).std(),
        "n_ngrams_match_1": np.array(num_overlapping_words).mean(),
        "n_ngrams_match_2": np.array(num_overlapping_bigrams).mean(),
        "n_ngrams_match_3": np.array(num_overlapping_trigrams).mean(),
        "num_true_words": np.array(num_true_words).mean(),
        "num_pred_words": np.array(num_pred_words).mean(),
    }
    ############################################################
    # Compute BLEU and ROUGE scores
    metric_bleu = evaluate.load("sacrebleu")
    metric_rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    bleu_results = np.array(
        [
            metric_bleu.compute(predictions=[p], references=[r])["score"]
            for p, r in zip(predictions_str, references_str)
        ]
    )

    rouge_result = metric_rouge.compute(
        predictions=predictions_str, references=references_str
    )

    bertscore_results = bertscore.compute(predictions=predictions_str, references=references_str, lang="en")
    bertscore_results = np.array(bertscore_results['f1'])

    exact_matches = np.array(predictions_str) == np.array(references_str)
    gen_metrics = {
        "bleu_score": bleu_results.mean(),
        "bleu_score_sem": sem(bleu_results),
        "bleu_score_sd": bleu_results.std(),

        "rouge_score": rouge_result[
            "rouge1"
        ],  # ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']

        "exact_match": exact_matches.mean(),
        "exact_match_sem": sem(exact_matches),
        "exact_match_sd": exact_matches.std(),

        "bertscore_score": bertscore_results.mean(),
        "bertscore_score_sem": sem(bertscore_results),
        "bertscore_score_sd": bertscore_results.std(),
    }

    all_metrics = {**set_token_metrics, **gen_metrics}
    return all_metrics


@torch.no_grad()
def evaluate_by_task(
        task_name: str,
        target_emb_model_name: str,
        source_emb_model_name: str,
        aligner_model: torch.nn.Module,
        aligner_dir: str,
        batch_size: int = 64,
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
    elif task_name.startswith('text_inversion__'):
        data_to_invert = task_name.split('__')[1]
        # results_target = eval_on_text_inversion(
        #     text_encoder_model_name=target_emb_model_name,
        #     data_to_invert=data_to_invert,
        #     batch_size=batch_size,
        #     trained_aligner_dir=None,  # keep it this way, so we don't load other model
        # )
        # TODO run only if source model embedding dim == gtr's (768)
        results_source = eval_on_text_inversion(
            text_encoder_model_name=source_emb_model_name,
            data_to_invert=data_to_invert,
            batch_size=batch_size,
            trained_aligner_dir=aligner_dir
        )
        results_source_w_aligned = eval_on_text_inversion(
            text_encoder_model_name=source_emb_model_name,
            data_to_invert=data_to_invert,
            aligner_model=aligner_model,
            batch_size=batch_size,
            trained_aligner_dir=aligner_dir
        )
        results = dict(
            # pairs_of_target=results_target,
            pairs_of_source=results_source,  # TODO prevent misalignment
            pairs_of_source_w_aligned=results_source_w_aligned,
            data_to_invert=data_to_invert,
        )
    elif task_name == 'sts':
        # TODO measure the embedding quality
        pass
    else:
        raise ValueError(f"Unknown evaluation task: {task_name}")

    return results


if __name__ == '__main__':
    eval_on_cifar_100()
