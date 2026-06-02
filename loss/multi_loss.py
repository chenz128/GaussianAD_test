import torch.nn as nn
from . import OPENOCC_LOSS
from misc.tb_wrapper import WrappedTBWriter
if 'selfocc' in WrappedTBWriter._instance_dict:
    writer = WrappedTBWriter.get_instance('selfocc')
else:
    writer = None

@OPENOCC_LOSS.register_module()
class MultiLoss(nn.Module):

    def __init__(self, loss_cfgs):
        super().__init__()

        assert isinstance(loss_cfgs, list)
        self.num_losses = len(loss_cfgs)

        losses = []
        for loss_cfg in loss_cfgs:
            losses.append(OPENOCC_LOSS.build(loss_cfg))
        self.losses = nn.ModuleList(losses)
        self.iter_counter = 0

    def forward(self, inputs):

        loss_dict = {}
        tot_loss = 0.
        for loss_func in self.losses:
            result = loss_func(inputs)
            if isinstance(result, tuple):
                loss, sub_dict = result
                tot_loss += loss
                loss_dict.update(sub_dict)
                if writer and self.iter_counter % 10 == 0:
                    for k, v in sub_dict.items():
                        writer.add_scalar(f'loss/{k}', v, self.iter_counter)
            else:
                loss = result
                tot_loss += loss
                loss_dict.update({
                    loss_func.__class__.__name__: \
                    loss.detach().item()
                })
                if writer and self.iter_counter % 10 == 0:
                    writer.add_scalar(
                        f'loss/{loss_func.__class__.__name__}',
                        loss.detach().item(), self.iter_counter)
        if writer and self.iter_counter % 10 == 0:
            writer.add_scalar(
                'loss/total', tot_loss.detach().item(), self.iter_counter)
        self.iter_counter += 1

        return tot_loss, loss_dict
