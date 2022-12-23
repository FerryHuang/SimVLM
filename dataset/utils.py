from transformers import AutoTokenizer

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, ConcatDataset

from dataset.coco_caption_dataset import CocoCaptionKarpathyDataset
from dataset.vg_caption_dataset import VisualGenomeCaptionDataset

DATASETS = {
    'coco': CocoCaptionKarpathyDataset,
    'vg': VisualGenomeCaptionDataset
}

TOKENIZER_MAP = {
    'BartTokenizer': 'facebook/bart-base',
    'BertGenerationTokenizer': 'google/bert_for_seq_generation_L-24_bbc_encoder',
    'RobertaTokenizer': 'roberta-base'
}

def create_tokenizer(tokenizer_name):
    """Creates a seq2seq tokenizer using a pretrained model from Hugging Face.

    Args:
        tokenizer_name: The name of the tokenizer to create. Must be one of:
            - 'BartTokenizer': Bart base model
            - 'BertGenerationTokenizer': Bert for sequence generation base model
            - 'RobertaTokenizer': RoBERTa base model

    Returns:
        A seq2seq tokenizer object.
    """
    

    if not tokenizer_name in TOKENIZER_MAP:
        raise ValueError("Invalid seq2seq tokenizer name")

    return AutoTokenizer.from_pretrained(TOKENIZER_MAP[tokenizer_name])


def create_dataset(
    dataset_names=['coco', 'vg'],
    split='train',
    **kwargs
):
    datasets = [DATASETS[dataset_name](split, **kwargs) for dataset_name in dataset_names]
    mixed_dataset = ConcatDataset(datasets)
    return mixed_dataset


def collate_batch(
    batch,
    bos_token_id,
    eos_token_id,
):  
    images, prefix_texts, decoder_input_texts, label_texts, groundtruths= [], [], [], [], []
    for image, prefix_text, tgt_text, groundtruh in batch:
        images.append(image)
        prefix_texts.append(prefix_text)
        # drop eos in decoder input and bos in label
        decoder_input_texts.append(
            tgt_text[tgt_text != eos_token_id]
        )
        label_texts.append(
            tgt_text[tgt_text != bos_token_id]
        )
        groundtruths.append(groundtruh)

    batch_items = {
        'image': torch.stack(images),
        'prefix_text': torch.stack(prefix_texts),
        'decoder_input_text': torch.stack(decoder_input_texts),
        'label_text': torch.stack(label_texts),
        'groundtruth': groundtruths
    }

    return batch_items
    

def create_dataloader(
    batch_size: int,
    dataset: Dataset,
    split='train',
    sampler=None,
    num_workers=4,
    pin_memory=True
):

    def collate_fn(batch):
        if isinstance(dataset, ConcatDataset):
            bos_token_id = getattr(dataset.datasets[0].processor.tokenizer, 'bos_token_id')
            eos_token_id = getattr(dataset.datasets[0].processor.tokenizer, 'eos_token_id')
        return collate_batch(
            batch,
            bos_token_id,
            eos_token_id
        )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler if sampler else RandomSampler(dataset),
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=(split == 'train')
    )
    return loader
    