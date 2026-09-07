from re import L
from typing import Tuple
import torch
import logging

from rich.logging import RichHandler


def setup_logging(log_file='log.txt', level=logging.INFO):
    """Setup logging configuration
    """
    logging.basicConfig(level=level,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')
    console = RichHandler()
    console.setLevel(level)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


__optimizers = {
    'SGD': torch.optim.SGD,
    'ASGD': torch.optim.ASGD,
    'Adam': torch.optim.Adam,
    'AdamW': torch.optim.AdamW,
    'Adamax': torch.optim.Adamax,
    'Adagrad': torch.optim.Adagrad,
    'Adadelta': torch.optim.Adadelta,
    'Rprop': torch.optim.Rprop,
    'RMSprop': torch.optim.RMSprop
}

# xxx_param_group: 一个tuple, 格式是(参数: list, 学习率: float)
def _get_optimizer_hyper_params(
    optimizer_name: str,
    clip_param_group: Tuple[list, float],
    lstm_param_group: Tuple[list, float],
    other_param_group: Tuple[list, float],
    momentum: float,
    enable_SGD_momentum,  # 设置为False时表示不对SGD使用动量
    weight_decay,
    alpha   # RMSProp参数, 用于控制移动指数平均的平滑常数和衰减率
):
    optimizer_grouped_hyper_params = {
        'AdamW': [
            {'params': clip_param_group[0], 'lr': clip_param_group[1], 'momentum': momentum,
                            'weight_decay': weight_decay},
            {'params': lstm_param_group[0], 'lr': lstm_param_group[1], 'momentum': momentum,
                            'weight_decay': weight_decay},
            {'params': other_param_group[0], 'lr': other_param_group[1], 'momentum': momentum,
                            'weight_decay': weight_decay}
        ],
        'Adam': [
            {'params': clip_param_group[0], 'lr': clip_param_group[1], 'momentum': momentum },
            {'params': lstm_param_group[0], 'lr': lstm_param_group[1], 'momentum': momentum },
            {'params': other_param_group[0], 'lr': other_param_group[1], 'momentum': momentum}
        ],
        'SGD': [
            {'params': clip_param_group[0], 'lr': clip_param_group[1], 'momentum': momentum if enable_SGD_momentum else 0.0,
                            'weight_decay': weight_decay},
            {'params': lstm_param_group[0], 'lr': lstm_param_group[1], 'momentum': momentum if enable_SGD_momentum else 0.0,
                            'weight_decay': weight_decay},
            {'params': other_param_group[0], 'lr': other_param_group[1], 'momentum': momentum if enable_SGD_momentum else 0.0,
                            'weight_decay': weight_decay}
        ],
        'RMSprop': [
            {'params': clip_param_group[0], 'lr': clip_param_group[1], 'momentum': momentum, 'alpha': alpha },
            {'params': lstm_param_group[0], 'lr': lstm_param_group[1], 'momentum': momentum, 'alpha': alpha },
            {'params': other_param_group[0], 'lr': other_param_group[1], 'momentum': momentum, 'alpha': alpha}
        ],
        'default': [
            {'params': clip_param_group[0], 'lr': clip_param_group[1], 'momentum': momentum },
            {'params': lstm_param_group[0], 'lr': lstm_param_group[1], 'momentum': momentum },
            {'params': other_param_group[0], 'lr': other_param_group[1], 'momentum': momentum}
        ]
    }

    if __optimizers.get(optimizer_name) != None and optimizer_grouped_hyper_params.get(optimizer_name) != None:
        return optimizer_grouped_hyper_params[optimizer_name]
    else:
        return optimizer_grouped_hyper_params['default']


def select_optimizer(optimizer_name, clip_param_group, lstm_param_group, other_param_group, *kargs, **kwargs):

    momentum = kwargs.get('momentum', 0.9)
    enable_SGD_momentum = kwargs.get('enable_sgd_momentum', True if optimizer_name =='SGD' else False)
    weight_decay = kwargs.get('weight_decay', 1e-4)
    alpha = kwargs.get('alpha', 0.99)

    try:
        if __optimizers.get(optimizer_name) == None:
            print(f"警告: 优化器{optimizer_name}未经预先实验, 可能产生意想不到的效果, 慎重使用。如果需要该优化器, 请手动添加到优化器字典中。")
            print(f"可用优化器: {__optimizers.keys()}")
        optimizer = __optimizers[optimizer_name](_get_optimizer_hyper_params(optimizer_name=optimizer_name,
                                                                    clip_param_group=clip_param_group,
                                                                    lstm_param_group=lstm_param_group,
                                                                    other_param_group=other_param_group,
                                                                    momentum=momentum,
                                                                    enable_SGD_momentum=enable_SGD_momentum,
                                                                    weight_decay=weight_decay,
                                                                    alpha=alpha)
                                                )
        return optimizer

    except KeyError as e:
        print(f"优化器{optimizer_name}不存在, 请查看https://docs.pytorch.org/docs/stable/optim.html#algorithms查看有哪些Optimizer可用")
        print("训练已退出")
        exit()


def adjust_optimizer(optimizer, epoch, config):
    """Reconfigures the optimizer according to epoch and config dict"""
    def modify_optimizer(optimizer, setting):
        for param_group in optimizer.param_groups:
            for key in param_group.keys():
                if key in setting:
                    logging.debug('OPTIMIZER - setting %s = %s' %
                                  (key, setting[key]))
                    param_group[key] = setting[key]
        return optimizer

    if callable(config):
        optimizer = modify_optimizer(optimizer, config(epoch))
    else:
        for e in range(epoch):  # run over all epochs - sticky setting
            if e in config:
                optimizer = modify_optimizer(optimizer, config[e])

    return optimizer


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
