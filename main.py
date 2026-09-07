import argparse
import os
import time

import logging
import torch.nn as nn
import torch.backends.cudnn as cudnn

import math
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import Optimizer
from data import build_vocab, get_coco_data, get_iterator
from model import CaptionClipModel
from utils import setup_logging, AverageMeter, select_optimizer
from torchvision.models import resnet

from torch.cuda.amp import autocast, GradScaler

import torch

import numpy as np
import random

from rich.console import Console

console = Console()

model_names = sorted(name for name in resnet.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(resnet.__dict__[name]))

parser = argparse.ArgumentParser(description='COCO caption genration training')

# 模型参数: 使用Vision Transformer还是Resnet作为CLIP的图像编码器部分
# 具体可在python控制台中输入`import clip;clip.available_models()`来查看可用的模型, 并在网络上根据名称搜索模型的特点
parser.add_argument('--encoder_model', '-e', default='ViT-B/32', type=str, help='name of image encoder model to be used', choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'])
# 模型参数: 设置词嵌入向量的维度大小为256, 表示使用256个数字来表示一个词token
parser.add_argument('--embedding_size', default=256, type=int,
                    help='size of word embedding used')
# 模型参数: RNN部分, 设置RNN一个隐状态存储的特征数量, 过高可能可以表示更丰富的信息, 但可能稀释记忆; 过低则可能无法学到有效的隐状态
parser.add_argument('--rnn_size', default=256, type=int,
                    help='size of rnn hidden layer')
# 模型参数: RNN部分, 设置RNN的层数。如果大于1, 则会形成堆叠LSTM, 后面的LSTM就会使用前面的LSTM的输出作为输入, 进一步计算隐状态并得到最终输出
parser.add_argument('--num_layers', default=2, type=int,
                    help='number of rnn layers to use')
# 模型参数: RNN部分, 设置一层LSTM输入的最大时间步, 决定数据集中一个样本最终可以输进LSTM的词长度, 也间接决定生成时的句子长度
parser.add_argument('--max_length', default=30, type=int,
                    help='maximum time length to feed')
# 训练参数: 张量的类型, 决定训练数据精度和数据类型, 不同精度对资源需求不同, 效果也不同。决定精度需量力而行, 可参考: docs.pytorch.org/docs/stable/tensors.html
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
# 训练参数: 微调的阶段数, 决定解冻的层数变化策略。默认为4, 表示每多一个阶段解冻的层数多25%
parser.add_argument('--finetune_stage', default=4, type=int,
                    help='finetune stages of CLIP, For progressive finetuning.')
# 训练参数: 开始训练的epoch位置。若不为0, 则从指定的epoch数继续训练。推荐于中断训练后重新开始时使用
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
# 训练参数: 优化器类型。推荐从SGD, AdamW, Adam和RMSProp中选择, 这些预设了机制可以指定特定的参数, 你也可以拓展更多的参数, 详见utils.py中的select_optimizer和_get_optimizer_hyper_params函数
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
# 训练参数: 梯度裁剪强度。更高的值有助于防止CLIP梯度爆炸, 但同时也会降低学习收敛速度
parser.add_argument('--grad_clip', default=6., type=float,
                    help='gradient max norm')
# 训练参数: 是否共享embedding层和classifer层的权重
parser.add_argument('--share_weights', default=False, type=bool,
                    help='share embedder and classifier weights')
# 训练参数: 数据加载线程数, 越高则加载数据的速度越快, 但更消耗资源。默认为0表示不使用多加载器
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 0)')
# 训练参数: 训练轮数, 不建议过小, CLIP的微调需要时间
parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
# 训练参数: 训练batch size, 可能需要修改: 因显存大小而异, 当报错显存不足(CUDA Out of Memory)时可以适当调小一点。建议是2的整数次方, 利于显存对齐, 提高训练效率
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 64)')
# 训练参数: 验证batch_size, 同上batch_size, 可能需要修改
parser.add_argument('-eb', '--eval_batch_size', default=32, type=int, metavar='N', help='mini-batch size (default: 32)')
# 训练参数: 基础学习率, 请确保学习率足够小, 因为我们涉及到微调CLIP这样的大规模预训练模型。这里的5e-3是除了CLIP以外模型的其他部分的学习率,
# CLIP的学习率会自动在这个基础上乘以1/200。
parser.add_argument('--lr', '--learning_rate', default=5e-3, type=float, metavar='LR', help='initial learning rate')
# 训练参数: 动量设置
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
# 训练参数: 权重衰减设置
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
# 训练参数: Alpha参数设置: 只会对RMSprop优化器生效
parser.add_argument('--alpha', default=0.99, type=float, help='smoothing constant for RMSprop')
# 训练参数: 是否对SGD优化器启用动量, 只有优化器是SGD并且momentum不是0时生效
parser.add_argument('--enable_sgd_momentum', default=True, type=bool, help='Enable momentum parameter for SGD optimizer, only effective when optimizer is set to SGD and momentum is not 0')
# 环境参数: 每隔多少个step打印一次训练过程信息
parser.add_argument('--print_freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
# 环境参数: 保存路径根目录
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
# 环境参数: 保存路径中用于存储模型的子文件夹
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
# 环境参数: 是否启用调试。如果传入这个标志, 则会启用调试输出, 包括神经网络输入输出维度等, 便于调试。
parser.add_argument('--debug', '-d', action='store_true', help='Enable Debugging')
# 环境参数: 要保留最近的检查点文件个数。由于检查点文件体积较大, 如果磁盘空间不够的话, 可以少保留一些。默认全部保留(设置为0, 负数也可)。
parser.add_argument('--keep_count', '-k', type=int, default=0, help='recent checkpoint files to keep')

def train_model(start_epoch, epochs,
                model: CaptionClipModel,
                optimizer: Optimizer,
                train_data, val_data,
                checkpoint_file: str,
                save_path: str,
                keep_count: int,
                vocab: list[str],
                finetune_stage: int,
                base_clip_lr: float,
                type_,
                print_freq: int,
                grad_clip: float):
    """
    训练模型。
    """

    use_cuda = 'cuda' in type_
    loss = nn.CrossEntropyLoss()
    perplexity = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # 内置函数方便访问变量
    def forward(model: CaptionClipModel, data, training=True, optimizer=None):

        scaler = torch.amp.GradScaler('cuda')

        timestamp = time.time()
        for batch_idx, (imgs, (captions, lengths)) in enumerate(data):
            data_time.update(time.time() - timestamp)
            if use_cuda:
                imgs = imgs.cuda()
                captions = captions.cuda()

            logging.debug(f"captions.size: {captions.size()}, lengths.size: {len(lengths)}")
            logging.debug(f"captions: {captions[:3]}")
            logging.debug(f"lengths: {lengths[:3]}")

            # 输入应该是标题序列的前n-1个词（不包括最后的 <EOS>）
            # 因为模型的目标是根据前面的词来预测下一个词。
            input_captions = captions[:, :-1]
            input_lengths = [l - 1 for l in lengths]

            # 目标应该是标题序列的后n-1个词（不包括最开始的词）
            # 我们希望模型在看到 w_i 后能预测出 w_{i+1}
            target_captions = captions[:, 1:]
            target_lengths = [l - 1 for l in lengths]

            logging.debug(f"target_captions:{target_captions.size()}")
            logging.debug(f"target_lengths: {target_lengths[:5]}")

            # 混合精度训练: 加速
            with torch.amp.autocast('cuda'):

                pred, _ = model(imgs, input_captions, input_lengths)

                # 打包目标序列
                packed_targets = pack_padded_sequence(target_captions, target_lengths, batch_first=True)
                target_flat = packed_targets.data

                logging.debug(f"pred.size: {pred.size()}, target_flat.size: {target_flat.size()}")

                err = loss(pred, target_flat)
                perplexity.update(math.exp(err.item()))

                if training:

                    if optimizer is None:
                        raise ValueError("Optimizer: 优化器不能为None")

                    optimizer.zero_grad()
                    scaler.scale(err).backward()
                    clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()

            batch_time.update(time.time() - timestamp)
            timestamp = time.time()

            if batch_idx % print_freq == 0:
                logging.info('{phase} - Epoch: [{0}][{1}/{2}]  '
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                             'Data {data_time.val:.3f} ({data_time.avg:.3f})  '
                             'Perplexity {perp.val:.4f} ({perp.avg:.4f})'.format(
                                 epoch, batch_idx, len(data),
                                 phase='TRAINING' if training else 'EVALUATING',
                                 batch_time=batch_time,
                                 data_time=data_time, perp=perplexity))

        return perplexity.avg


    for epoch in range(start_epoch, epochs):

        model.train()
        stage = model.finetune_clip(optimizer=optimizer,
                                    current_epoch=epoch,
                                    total_epochs=epochs,
                                    base_clip_lr=base_clip_lr,
                                    stages=finetune_stage)

        # 训练
        train_perp = forward(
            model, train_data, training=True, optimizer=optimizer)

        model.eval()
        # 评估
        val_perp = forward(model, val_data, training=False)

        logging.info('\n Epoch: {0}\t'
                     'Training Perplexity {train_perp:.4f} \t'
                     'Validation Perplexity {val_perp:.4f} \n'
                     .format(epoch + 1, train_perp=train_perp, val_perp=val_perp))

        model.save_checkpoint(checkpoint_file % (epoch + 1), save_path, keep_count, epoch)

def set_seed(seed=42):
    """设置所有随机数生成器的种子以确保复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    console.log(f"随机种子已设置为: {seed}", style="green")

def main():
    set_seed()

    args = parser.parse_args()
    save_path = os.path.join(args.results_dir, args.save)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'), level=logging.DEBUG if args.debug else logging.INFO)
    checkpoint_file = os.path.join(save_path, 'checkpoint_epoch_%s.pth')

    logging.debug("训练参数: %s", args)

    use_cuda = ('cuda' in args.type) and torch.cuda.is_available()

    vocab = build_vocab()

    model = CaptionClipModel(vocab, args.encoder_model,
                         embedding_size=args.embedding_size,
                         rnn_size=args.rnn_size,
                         num_layers=args.num_layers,
                         share_embedding_weights=args.share_weights,
                         use_cuda=use_cuda)

    train_data = get_iterator(get_coco_data(vocab, train=True),
                              batch_size=args.batch_size,
                              max_length=args.max_length,
                              shuffle=True,
                              num_workers=args.workers)

    val_data = get_iterator(get_coco_data(vocab, train=False),
                            batch_size=args.eval_batch_size,
                            max_length=args.max_length,
                            shuffle=False,
                            num_workers=args.workers)


    if use_cuda:
        cudnn.benchmark = True
        model.cuda()

    clip_lr = args.lr * 0.005  # CLIP的学习率是基础的1 / 200
    lstm_lr = args.lr
    other_lr = args.lr

    clip_params = []
    lstm_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith('clip_model.'):
            clip_params.append(param)
        elif name.startswith('rnn.'):
            lstm_params.append(param)
        else:
            other_params.append(param)

    optimizer = select_optimizer(args.optimizer,
                                clip_param_group=(clip_params, clip_lr),
                                lstm_param_group=(lstm_params, lstm_lr),
                                other_param_group=(other_params, other_lr),
                                **{'momentum': args.momentum,
                                'alpha': args.alpha,
                                'enable_sgd_momentum': args.enable_sgd_momentum,
                                'weight_decay': args.weight_decay}
    )

    finetune_stage = args.finetune_stage

    console.log(f"使用设备:{next(model.parameters()).device}", style="bold blue")

    train_model(args.start_epoch,
                args.epochs,
                model,
                optimizer,
                train_data,
                val_data,
                checkpoint_file,
                save_path,
                args.keep_count,
                vocab,
                finetune_stage,
                base_clip_lr=clip_lr,
                type_=args.type,
                print_freq=args.print_freq,
                grad_clip=args.grad_clip
    )

if __name__ == '__main__':
    main()
