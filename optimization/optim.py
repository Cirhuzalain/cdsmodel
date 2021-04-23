import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Optimization(object):

    def __init__(self, params, lr, hidden_size, warm_up, n_step):
        self.original_lr = lr
        self.n_step = n_step
        self.hidden_size = hidden_size
        self.warm_up_step = warm_up
        self.weight_decay = 0.07368935714782943
        self.lr_schedule_patience = 1
        self.lr_schedule_factor = 0.18069313890040145
        self.optimizer = optim.AdamW(params, betas=[0.9, 0.998], eps=1e-9, lr=lr, weight_decay=self.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=self.lr_schedule_factor, patience=self.lr_schedule_patience, verbose=True)

    def step(self):
        self.n_step += 1
        self.optimizer.step()


class LabelSmoothing(nn.Module):

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index):
        super(LabelSmoothing, self).__init__()
        self.padding_idx = ignore_index
        self.label_smoothing = label_smoothing
        self.vocab_size = tgt_vocab_size
        

    def forward(self, output, target):
        target = target[:, 1:].contiguous().view(-1)
        output = output.view(-1, self.vocab_size)
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -output.gather(dim=-1, index=target.view(-1, 1))[non_pad_mask].sum()
        smooth_loss = -output.sum(dim=-1, keepdim=True)[non_pad_mask].sum()
        eps_i = self.label_smoothing / self.vocab_size
        loss = (1. - self.label_smoothing) * nll_loss + eps_i * smooth_loss
        return loss / non_pad_mask.float().sum()