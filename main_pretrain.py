import wandb

import os
from typing import Iterable
from pathlib import Path
import time
import datetime
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DistributedSampler, RandomSampler


from simvlm import SimVLMModel, SimVLMConfig, SimVLMProcessor
from dataset import create_tokenizer, create_dataset, create_dataloader, vit_transform_randaug
import utils.dist_utils as dist_utils
import utils.misc as misc


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens proces

def train_one_epoch(
    device: torch.device,
    dataloader: Iterable,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_compute,
    accum_iter=1,
    train_state=TrainState(),
    run=None,
    do_log=False,
):
    """Train a single epoch"""
    model.train(True)
    optimizer.zero_grad()

    start = time.time()
    total_tokens = 0
    batch_losses = []
    tokens = 0
    n_accum = 0
    # TODO Track other traning states
    for iter, batch in enumerate(dataloader):
        image, prefix_text, decoder_input_text, label_text, _ = batch.values()
        image, prefix_text, decoder_input_text, label_text = (
            image.to(device), prefix_text.to(device), decoder_input_text.to(device), label_text.to(device)
        ) 
        loss = model(image, prefix_text, decoder_input_text, label_text)
        # loss = loss_compute(out, label_text)
        loss.backward()

        train_state.step += 1
        train_state.samples += image.shape[0]
        if iter % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            n_accum += 1
            train_state.accum_step += 1
        scheduler.step()

        if do_log:
            run.log({"batch_loss": float(loss.cpu()), 
                     "lr": float(optimizer.param_groups[0]["lr"])})
        batch_losses.append(float(loss.cpu()))

    return batch_losses, train_state

def main(args, run=None):
    # Check to see if local_rank is 0 and if log params
    is_master = (args.local_rank == 0) and dist_utils.is_main_process()
    do_log = run is not None

    #TODO config parameters like vocab_size, pad_token_id are set at the moment the tokenizer being set
    # so model(config) can only be built after dataset(config), this may be improved by refructuring codes 
    # on how config model and dataset
    config = SimVLMConfig()
    dist_utils.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + dist_utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    # prepare data
    dataset_train = create_dataset(
        dataset_names=args.dataset_names,
        split='train',
        processor=SimVLMProcessor(
                create_tokenizer('BartTokenizer'),
            ),
        transforms=vit_transform_randaug,
        config=config,
        seed=seed
    )

    if args.distributed:
        num_tasks = dist_utils.get_world_size()
        global_rank = dist_utils.get_rank()
        sampler_train = DistributedSampler(
            dataset=dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = RandomSampler(dataset_train)

    dataloader_train = create_dataloader(
        batch_size=args.batch_size,
        dataset=dataset_train,
        sampler=sampler_train,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    # define the model
    model = SimVLMModel(config=config)
    model.to(device)
    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * dist_utils.get_world_size()

    if args.lr is None: # only base_lr is specified
        args.lr = args.base_lr * eff_batch_size / 256

    print("base_lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual_lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[args.gpu]
        )
        model_without_ddp = model.module
    # watch gradients only for rank 0
    if is_master:
        run.watch(model)

    # optimizer
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # lr_scheduler ued in transformer
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: misc.rate(
            step, 768, factor=1, warmup_steps=args.warmup_steps
        ),
    )

    # criterion copied from http://nlp.seas.harvard.edu/annotated-transformer/#loss-computation
    criterion = misc.LabelSmoothing(config.vocab_size, config.pad_token_id, smoothing=0.1)
    loss_compute = misc.SimpleLossCompute(criterion=criterion)
    
    if is_master and run.resumed:
        checkpoint = torch.load(
            args.checkpoint_path,
            map_location='cpu'
        )
        model_without_ddp.load_state_dict(checkpoint['model']) # have to load weights onto model_WITHOUT_DDP
        args.start_epoch = checkpoint['epoch']
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    now = datetime.datetime.now()
    save_dir = os.path.join(args.output_dir, 'run-' + now.strftime('%Y-%m-%d %H:%M:%S'))
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            dataloader_train.sampler.set_epoch(epoch)
            batch_losses, train_stats = train_one_epoch(
                device, dataloader_train, model, optimizer, lr_scheduler, loss_compute,
                accum_iter=args.accum_iter, run=run, do_log=do_log
            )

            if epoch % args.print_every and is_master:
                print(f"Epoch{epoch}: loss = {batch_losses[-1]}")

            if args.output_dir and ((epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs):

                misc.save_model(
                    args, epoch, batch_losses[-1], model_without_ddp, optimizer, lr_scheduler=lr_scheduler, save_dir=save_dir
                )
                if is_master:
                    run.save(save_dir)
                

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SimVLM pre-training', add_help=False)
    # File root
    parser.add_argument('--data_root', default='', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='./result', type=str,
                        help='Path where to save outputs')
    parser.add_argument('--checkpoint_path', default='')

    # Optimizer params
    parser.add_argument('--lr', default=None, type=float, metavar='LR',
                        help='Learning rate')
    parser.add_argument('--base_lr', default=5*1e-4, type=float, metavar='LR',
                        help='Base learning rate')
    parser.add_argument('--warmup_steps', default=3000, type=int, metavar='N',
                        help='Steps to warm up lr')
    
    # Data
    parser.add_argument('--dataset_names', default=['coco'])
    parser.add_argument('--batch_size', default=16, type=int,
                        help="Batch size per gpu")
    parser.add_argument('--random_prefix_len', default=False, type=bool,
                        help='is prefix text length randomly chosen')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_memory', default=True, type=bool)
    
    # Training params
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations')
    parser.add_argument('--model', default='', type=str, metavar='MODEL',
                        help='Name of model to pretrain')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--print_every', default=1, type=int,
                        help='number of epochs of interval to save checkpoints')
    parser.add_argument('--save_every', default=10000, type=int,
                        help='number of epochs of interval to save checkpoints')
   
    # distributed training parameters
    parser.add_argument("--distributed", default=True, type=bool)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    # wandb
    parser.add_argument('--run_name', default='', type=str)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--run_id', default=None)

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.local_rank == 0 and dist_utils.is_main_process():
        run = wandb.init(
            project="simvlm",
            name=args.run_name,
            dir=args.output_dir,
            config={
                'datasets': args.dataset_names,
                "base_lr": args.base_lr,
                "warmup_steps": args.warmup_steps,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "accum_iter": args.accum_iter,
                "seed": args.seed,
                'random_prefix_len': args.random_prefix_len
            },
            resume=args.resume,
            id=args.run_id
        )
        args.run_id = run.id
    else:
        run = None

    main(args, run)
    # wandb.finish()