import json
import torch
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

from dataset import create_dataset, create_dataloader, create_tokenizer, vit_transform
from simvlm import SimVLMConfig, SimVLMModel, SimVLMProcessor

def main(args):
    config = SimVLMConfig()
    tokenizer = create_tokenizer('BartTokenizer')
    data_test = create_dataset(
        dataset_names=['coco'],
        split=['test'],
        processor=SimVLMProcessor(
            tokenizer
        ),
        transforms=vit_transform,
        config=config
    )

    dataloader_test = create_dataloader(
        batch_size=32,
        dataset=data_test,
        split='test',
    )

    model = SimVLMModel(config).load_state_dict(
        torch.load("result/checkpoint-0.pth", map_location=args.device)
    )
    
    generated_captions = []
    for batch_idx, image, _, _ in enumerate(dataloader_test):
        generated_caption = model.generate(image)
        generated_captions.append(generated_caption)

    tokenizer.batch_decode