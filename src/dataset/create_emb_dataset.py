import torch
import datasets
import huggingface_hub
from sentence_transformers import SentenceTransformer
import json
import os

def load_passages_from_dataset(dataset_name: str = "mteb/scifact") -> datasets.Dataset:
    if dataset_name == "scifact":
        # contains 'title' and 'text' columns
        ds = datasets.load_dataset("mteb/scifact", "corpus")
        ds = ds['corpus']
    elif dataset_name == 'coco_captions':
        # loads text captions of `annotations_trainval2014` from COCO dataset
        with open(os.path.join('data', 'annotations_trainval2014', 'annotations', 'captions_train2014.json')) as f:
            captions_dict = json.load(f)['annotations']
        # create an huggingface dataset
        ds = datasets.Dataset.from_list(captions_dict)
        ds = ds.rename_column('caption', 'text')
        ds = ds.rename_column('id', '_id')
        # TODO ensure ds has not training key or something
    else:
        raise NotImplementedError
    return ds


def get_repo_id(text_dataset_name: str, embedder_model_name: str) -> str:
    model_short_name = {
        "sentence-transformers/all-mpnet-base-v2": "ampnet",
        "sentence-transformers/all-MiniLM-L6-v2": "minilm-l6",
        "sentence-transformers/all-MiniLM-L12-v2": "minilm-l12",
        # "sentence-transformers/gtr-t": "gte-base-en-v1.5",
        'thenlper/gte-base': 'gte',
        'intfloat/e5-base-v2': 'e5',
        'sentence-transformers/clip-ViT-L-14': 'clip-text',
    }[embedder_model_name]
    return f"MatanBT/{text_dataset_name}__{model_short_name}"


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
    model = SentenceTransformer(embedder_model_name)
    # Embed (uses SeT to tokenize->forward-pass the text)
    dataset = dataset.map(lambda x: {'embedding': model.encode(x["text"], batch_size=batch_size,
                                                               normalize_embeddings=True)},
                          batched=True, batch_size=batch_size * 10)
    # Upload the dataset to the Hugging Face Hub
    dataset.push_to_hub(repo_id, private=True)
    print(f"Dataset uploaded to Hugging Face Hub: {repo_id}, with {len(dataset)} samples.")


def create_dataset_from_vision_encoder(

):
    # same as 'create_dataset', but accepts a vision encoder model
    pass  # TODO


class SourceTargetEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, text_dataset_name: str, source_emb_model_name: str, target_emb_model_name: str):
        self.text_dataset_name = text_dataset_name
        self.source_emb_model_name = source_emb_model_name
        self.target_emb_model_name = target_emb_model_name

        # Load data from HuggingFace
        source_repo_id = get_repo_id(text_dataset_name, source_emb_model_name)
        source_dataset = datasets.load_dataset(source_repo_id)
        source_dataset = (source_dataset['train']  #['corpus']  # TODO get rid of this 'corpus'
                          .sort("_id")  # ensures aligned order
                          .rename_column('embedding', 'emb_source')
                          .with_format('torch')
                          )
        self.source_dataset = source_dataset

        target_repo_id = get_repo_id(text_dataset_name, target_emb_model_name)
        target_dataset = datasets.load_dataset(target_repo_id)
        target_dataset = (target_dataset['train']  #['corpus'] TODO
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


if __name__ == '__main__':
    create_dataset(
        text_dataset_name="coco_captions",
        # embedder_model_name="sentence-transformers/clip-ViT-L-14",
        embedder_model_name="intfloat/e5-base-v2",
        # embedder_model_name="sentence-transformers/all-MiniLM-L12-v2",
    )
