# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from torch.optim.lr_scheduler import _LRScheduler


class PolynomialDecayLR(_LRScheduler):

    def __init__(self, optimizer, warmup_updates, tot_updates, lr, end_lr, power, last_epoch=-1, verbose=False):
        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.lr = lr
        self.end_lr = end_lr
        self.power = power
        super(PolynomialDecayLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            self.warmup_factor = self._step_count / float(self.warmup_updates)
            lr = self.warmup_factor * self.lr
        elif self._step_count >= self.tot_updates:
            lr = self.end_lr
        else:
            warmup = self.warmup_updates
            lr_range = self.lr - self.end_lr
            pct_remaining = 1 - (self._step_count - warmup) / (
                self.tot_updates - warmup
            )
            lr = lr_range * pct_remaining ** (self.power) + self.end_lr

        return [lr for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        assert False

if __name__=="__main__":
    import torch

    epochs=12
    batch_size=128
    peak_lr = 2e-4
    end_lr = 1e-9
    total_updates=33000*epochs/batch_size
    warmup_updates =total_updates // 10


    my_model=torch.nn.Linear(5,10)
    optimizer=torch.optim.AdamW(my_model.parameters(),lr=peak_lr,betas=(0.99,0.999))

    scheduler = PolynomialDecayLR(
        optimizer=optimizer,
        warmup_updates=warmup_updates,
        tot_updates=total_updates,
        lr=peak_lr,
        end_lr=end_lr,
        power=1
    )
    print(scheduler)
    for i in range(100):
        print(optimizer)
        scheduler.step()
