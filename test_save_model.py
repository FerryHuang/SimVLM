from unittest import TestCase
from argparse import ArgumentParser
from simvlm import SimVLMConfig, SimVLMModel
from utils.misc import save_model
from torch.optim import AdamW


    
model = SimVLMModel(SimVLMConfig())
parser = ArgumentParser()
parser.add_argument("--epochs", default=10)
parser.add_argument('--output_dir', default='./result', type=str,
                        help='Path where to save outputs')

args = parser.parse_args()
optimizer = AdamW(model.parameters())
save_model(args=args, epoch=0, model_without_ddp=model, optimizer=optimizer)