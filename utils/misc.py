from pathlib import Path
import os
import json
import time
import utils.dist_utils as dist_utils
import torch
import torch.nn as nn


def save_model(args, epoch, last_step_loss, 
                model_without_ddp: torch.nn.parallel.DistributedDataParallel,
                optimizer, lr_scheduler, loss_scaler=None, save_dir=None):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    time_tag = str(int(time.time()))

    to_save = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'loss': last_step_loss,
        'lr_scheduler': lr_scheduler.state_dict(),
    }
    checkpoint_path = os.path.join(save_dir, '-checkpoint-epoch%s.pth' % epoch_name)
    args_json_path = os.path.join(save_dir, 'args.json')

    if loss_scaler:
        to_save.update({'scaler': loss_scaler.state_dict()})

    dist_utils.save_on_master(to_save, checkpoint_path)
    if dist_utils.is_main_process():
        if not os.path.exists(args_json_path):
            with open(args_json_path, 'wt') as f:
                json.dump(vars(args), f, indent=4)



def rate(step, model_size, factor, warmup_steps):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    step = 1 if step == 0 else step

    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
    )



class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, criterion):
        self.criterion = criterion

    # def __call__(self, x, y, norm):
        # sloss = (
        #     self.criterion(
        #         x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
        #     )
        #     / norm
        # )
        # return sloss.data * norm, sloss
    def __call__(self, x, y):
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)),
                y.contiguous().view(-1)
            )
        )
        return sloss