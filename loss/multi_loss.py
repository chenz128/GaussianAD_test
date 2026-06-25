import torch.nn as nn
from . import OPENOCC_LOSS
from misc.tb_wrapper import WrappedTBWriter
if 'selfocc' in WrappedTBWriter._instance_dict:
    writer = WrappedTBWriter.get_instance('selfocc')
else:
    writer = None

@OPENOCC_LOSS.register_module()
class MultiLoss(nn.Module):

    def __init__(self, loss_cfgs, group_map=None):
        super().__init__()

        assert isinstance(loss_cfgs, list)
        self.num_losses = len(loss_cfgs)

        losses = []
        for loss_cfg in loss_cfgs:
            losses.append(OPENOCC_LOSS.build(loss_cfg))
        self.losses = nn.ModuleList(losses)
        self.iter_counter = 0
        # group_map: {loss_class_name: 'main'|'aux'} for PCGrad gradient surgery.
        # When set, forward also accumulates per-group loss tensors (grad kept)
        # into self.group_losses so the training loop can do PCGrad.
        self.group_map = group_map
        self.group_losses = None

    def forward(self, inputs):

        loss_dict = {}
        tot_loss = 0.
        # per-group loss tensors (keep grad) for PCGrad
        group_sums = {} if self.group_map is not None else None
        for loss_func in self.losses:
            cls_name = loss_func.__class__.__name__
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
                    cls_name: \
                    loss.detach().item()
                })
                if writer and self.iter_counter % 10 == 0:
                    writer.add_scalar(
                        f'loss/{cls_name}',
                        loss.detach().item(), self.iter_counter)
            if group_sums is not None:
                grp = self.group_map.get(cls_name, 'main')
                if grp not in group_sums:
                    group_sums[grp] = loss
                else:
                    group_sums[grp] = group_sums[grp] + loss
        if writer and self.iter_counter % 10 == 0:
            writer.add_scalar(
                'loss/total', tot_loss.detach().item(), self.iter_counter)
        self.iter_counter += 1
        self.group_losses = group_sums

        return tot_loss, loss_dict
