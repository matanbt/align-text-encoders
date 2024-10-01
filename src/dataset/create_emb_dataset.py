from functools import partial
from typing import List, Callable

import torch
import datasets
import huggingface_hub
from sentence_transformers import SentenceTransformer
import json
import os

from transformers import AutoTokenizer


def load_passages_from_dataset(dataset_name: str = "mteb/scifact") -> datasets.Dataset:
    # Textual datasets:
    if dataset_name == "scifact":
        # contains 'title' and 'text' columns
        ds = datasets.load_dataset("mteb/scifact", "corpus")
        ds = ds['corpus']  # TDO make sure `corpus` is not a key
    elif dataset_name == 'nq-corpus':
        ds = datasets.load_dataset("jxm/nq_corpus_dpr")
        ds['validation'] = ds['dev']
        del ds['dev']
        for split in ds.keys():
            ds[split] = ds[split].add_column('_id', list(range(len(ds[split]))))  # create an id column, by enumerating the dataset
    # Image-caption datasets:
    elif dataset_name == 'coco_captions':
        # loads text captions of `annotations_trainval2014` from COCO dataset
        with open(os.path.join('data', 'annotations_trainval2014', 'annotations', 'captions_train2014.json')) as f:
            captions_dict = json.load(f)['annotations']
        ds = datasets.Dataset.from_list(captions_dict)
        ds = ds.rename_column('caption', 'text')
        ds = ds.rename_column('id', '_id')

        with open(os.path.join('data', 'annotations_trainval2014', 'annotations', 'captions_val2014.json')) as f:
            captions_dict = json.load(f)['annotations']
        val_ds = datasets.Dataset.from_list(captions_dict)
        val_ds = val_ds.rename_column('caption', 'text')
        val_ds = val_ds.rename_column('id', '_id')

        # map each split to a dataset
        ds = datasets.DatasetDict({'train': ds, 'validation': val_ds})
    elif dataset_name == 'conc_captions':
        ds = datasets.load_dataset("google-research-datasets/conceptual_captions", "unlabeled")
        for split in ds.keys():
            ds[split] = ds[split].rename_column('caption', 'text')
            ds[split] = ds[split].add_column('_id', list(range(len(ds[split]))))    # create an id column, by enumerating the dataset
    else:
        raise NotImplementedError
    return ds


def get_repo_id(text_dataset_name: str, embedder_model_name: str) -> str:
    model_short_name = {
        "sentence-transformers/all-mpnet-base-v2": "ampnet",
        "sentence-transformers/all-MiniLM-L6-v2": "minilm-l6",
        "sentence-transformers/all-MiniLM-L12-v2": "minilm-l12",
        'thenlper/gte-base': 'gte',
        'sentence-transformers/gtr-t5-base': 'gtr-unnorm',
        'intfloat/e5-base-v2': 'e5',
        'sentence-transformers/clip-ViT-L-14': 'clip-text',
        'openai/clip-vit-large-patch14': 'clipl14-text',
        'sentence-transformers/average_word_embeddings_glove.6B.300d': 'glove',
        'Snowflake/snowflake-arctic-embed-m': 'arctic',
    }[embedder_model_name]
    return f"MatanBT/{text_dataset_name}__{model_short_name}"


def get_text_enc_function(embedder_model_name: str, device: str = 'cuda') -> Callable[[List[str]], torch.Tensor]:
    if embedder_model_name.startswith("openai/clip"):  # e.g. `openai/clip-vit-large-patch14`
        from transformers import CLIPModel
        model = CLIPModel.from_pretrained(embedder_model_name)
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(embedder_model_name, return_tensors="pt")

        def embed_batch(batch):
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            emb = model.get_text_features(**inputs)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            return emb
        return embed_batch
    elif embedder_model_name == 'sentence-transformers/gtr-t5-base':
        return gtr_t5_enc_from_vec2text()
    elif embedder_model_name == 'avg-glove-6B-300d':
        model = SentenceTransformer("average_word_embeddings_glove.6B.300d", device=device)
        return lambda x: model.encode(x, normalize_embeddings=True, convert_to_tensor=True)
    else:  # assumes it's a text encoder, available in SentenceTransformers
        model = SentenceTransformer(embedder_model_name, device=device)
        return lambda x: model.encode(x, normalize_embeddings=True, convert_to_tensor=True)


def create_dataset(
        text_dataset_name: str,
        embedder_model_name: str,
        n_passage_limit: int = torch.inf,  # DEBUG TODO comment
        skip_if_repo_exists: bool = False,
        batch_size=256,
):
    """
    Creates a dataset with a column of 'text' and a column of (the corresponding) 'embedding',
    then it uploads to the Hugging Face Hub.
    """
    repo_id = get_repo_id(text_dataset_name, embedder_model_name)

    # Check if dataset exists in HuggingFace hub
    if skip_if_repo_exists:
        try:
            repo_info = huggingface_hub.dataset_info(repo_id)
            print(f"Dataset already exists in Hugging Face Hub: {repo_id}. \n >> {repo_info}")
            return
        except huggingface_hub.utils.RepositoryNotFoundError:
            pass  # continues

    # Load the dataset
    dataset = load_passages_from_dataset(text_dataset_name)

    # limit the embedded passages to a given limit
    if len(dataset) > n_passage_limit:
        dataset = dataset.select(range(n_passage_limit))

    # Load the model
    embed_batch = get_text_enc_function(embedder_model_name)
    dataset = dataset.map(lambda x: {'embedding': embed_batch(x['text'])}, batched=True, batch_size=batch_size)

    # Upload the dataset to the Hugging Face Hub
    dataset.push_to_hub(repo_id, private=True)
    print(f"Dataset uploaded to Hugging Face Hub: {repo_id}, with {len(dataset)} samples.")


def create_dataset_from_vision_encoder(

):
    # same as 'create_dataset', but accepts a vision encoder model
    pass  # TODO


class SourceTargetEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self,
                 text_dataset_name: str, source_emb_model_name: str, target_emb_model_name: str,
                 train: bool = True):
        self.text_dataset_name = text_dataset_name
        self.source_emb_model_name = source_emb_model_name
        self.target_emb_model_name = target_emb_model_name
        self.split_name = 'train' if train else 'validation'

        # Load data from HuggingFace
        source_repo_id = get_repo_id(text_dataset_name, source_emb_model_name)
        source_dataset = datasets.load_dataset(source_repo_id)
        source_dataset = (source_dataset[self.split_name]
                          .sort("_id")  # ensures aligned order
                          .rename_column('embedding', 'emb_source')
                          .with_format('torch')
                          )
        self.source_dataset = source_dataset

        target_repo_id = get_repo_id(text_dataset_name, target_emb_model_name)
        target_dataset = datasets.load_dataset(target_repo_id)
        target_dataset = (target_dataset[self.split_name]
                          .sort("_id")  # ensures aligned order
                          .rename_column('embedding', 'emb_target')
                          .with_format('torch')
                          )
        self.target_dataset = target_dataset

        # Ensure datasets are aligned
        assert len(source_dataset) == len(target_dataset), "Datasets must have the same length"
        assert len(set(source_dataset['_id'])) == len(source_dataset), "The `_id` column must be unique"
        assert (source_dataset[52]['_id'] == target_dataset[52]['_id']), "Datasets must be aligned"
        assert (source_dataset[52]['text'] == target_dataset[52]['text']), "Datasets must be aligned"

    def __len__(self):
        return len(self.source_dataset)

    def __getitem__(self, idx):
        return self.source_dataset[idx]['emb_source'], self.target_dataset[idx]['emb_target']

    def get_metadata(self):
        return {
            'text_dataset_name': self.text_dataset_name,
            'source_emb_model_name': self.source_emb_model_name,
            'target_emb_model_name': self.target_emb_model_name,

            'source_emb_dim': self.source_dataset[0]['emb_source'].shape[0],
            'target_emb_dim': self.target_dataset[0]['emb_target'].shape[0],
        }


def gtr_t5_enc_from_vec2text():
    """Simply GTR-T5-Base, but without the final normalization (L2), as used in Vec2Text."""
    import torch
    from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel

    def mean_pool(
            hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        B, S, D = hidden_states.shape
        unmasked_outputs = hidden_states * attention_mask[..., None]
        pooled_outputs = unmasked_outputs.sum(dim=1) / attention_mask.sum(dim=1)[:, None]
        assert pooled_outputs.shape == (B, D)
        return pooled_outputs

    def get_gtr_embeddings(text_list,
                           encoder: PreTrainedModel,
                           tokenizer: PreTrainedTokenizer) -> torch.Tensor:
        inputs = tokenizer(text_list,
                           return_tensors="pt",
                           max_length=128,
                           truncation=True,
                           padding="max_length").to("cuda")

        with torch.no_grad():
            model_output = encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            hidden_state = model_output.last_hidden_state

            embeddings = mean_pool(hidden_state, inputs['attention_mask'])

        return embeddings

    encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")

    return partial(get_gtr_embeddings, encoder=encoder, tokenizer=tokenizer)


if __name__ == '__main__':
    create_dataset(
        # text_dataset_name="conc_captions",
        text_dataset_name="nq-corpus",
        # text_dataset_name="conc_captions",
        # embedder_model_name="sentence-transformers/clip-ViT-L-14",
        # embedder_model_name="openai/clip-vit-large-patch14",
        embedder_model_name="intfloat/e5-base-v2",
        # embedder_model_name="sentence-transformers/all-MiniLM-L12-v2",
    )
