import torch.nn as nn
import torch
import clip

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import transforms

from beam_search import CaptionGenerator

from torch import Tensor

import logging
import math

import os

class CaptionClipModel(nn.Module):

    def __init__(self, vocab: list[str], clip_model="ViT-B/32", embedding_size=512, rnn_size=512, num_layers=2, share_embedding_weights=False, use_cuda=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.vocab = vocab

        self.encoder_name = clip_model

        self.clip_model, self.preprocess = clip.load(clip_model, device='cuda' if use_cuda else "cpu", download_root="./models")

        # clip embedding层的维度, 用于传入后续的线性层映射到高维空间
        clip_embedding_dim: int = self.clip_model.visual.output_dim

        # 投影层: clip输出 -> embedding空间
        self.projection = nn.Linear(clip_embedding_dim, embedding_size)

        # LSTM: embedding空间 -> 词向量
        self.rnn = nn.LSTM(embedding_size, rnn_size, num_layers=num_layers)

        # 分类器
        self.classifier = nn.Linear(rnn_size, len(vocab))

        # 词嵌入层
        self.embedder = nn.Embedding(len(self.vocab), embedding_size)

        # Dropout
        self.dropout = nn.Dropout(0.2)

        # 共享权重(如果需要)
        if share_embedding_weights:
            self.embedder.weight = self.classifier.weight

        # 标准化参数
        self.normalize_values = {"mean": [0.48145466, 0.4578275, 0.40821073],
                                "std": [0.26862954, 0.26130258, 0.27577711]}

        # 默认是冻结状态
        for param in self.clip_model.parameters():
            param.requires_grad = False


    def forward(self, imgs: Tensor, captions: Tensor, lengths) -> tuple[Tensor, Tensor]:
        # clip: 图像 -> 编码后的图像特征
        img_features = self.clip_model.encode_image(imgs)

        # 从float16(Half类型) 转换成 float32, 因为后面的投影层需要float32。不建议将投影层的精度改为float16, 因为会影响训练稳定性(精度低)。
        img_features = img_features.float()

        # 投影层: 编码后的图像特征 -> 高维图像特征
        img_features = self.projection(img_features)

        # 将维度从(batch_size, embed_dim) 拓展成 (batch_size, 1, embed_dim), 便于后续拼接到caption的embedding序列头部，作为第一个"词"
        img_features_expanded = img_features.unsqueeze(1)

        # 更新长度反映多了一个图像特征
        updated_lengths = [l + 1 for l in lengths]

        # 嵌入层: 文本 -> 编码后的文本特征
        embeddings = self.embedder(captions)

        logging.debug(f"嵌入层形状: {embeddings.shape}")

        # Dropout层: 防止过拟合
        embeddings = self.dropout(embeddings)

        # 执行拼接
        fused_embeddings = torch.cat([img_features_expanded, embeddings], dim=1)

        logging.debug(f"拼接后形状: {embeddings.shape}")

        # 打包对齐
        packed_embeddings = pack_padded_sequence(fused_embeddings, updated_lengths, batch_first=True)

        # LSTM: 嵌入向量 -> 输出, 隐状态
        feats, state = self.rnn(packed_embeddings)

        # 解包
        unpacked_feats, _ = pad_packed_sequence(feats, batch_first=True)

        # 去掉第一个时间步的输出, 因为是图像特征, 不是我们想要的
        pred_input = unpacked_feats[:, 1:, :].contiguous()

        # 裁剪多余的, 对齐回来
        cropped_packed_outputs = pack_padded_sequence(pred_input, lengths, batch_first=True)

        logging.debug(f"RNN输出形状: {cropped_packed_outputs}")

        # 分类器: 输出 -> 词概率
        pred: Tensor = self.classifier(cropped_packed_outputs.data)

        logging.debug(f"分类器输出形状: {pred.shape}")

        # 返回词概率和隐状态
        return pred, state


    def generate(self, img, beam_size=3, max_caption_length=30,
                 length_normalization_factor=0.0, eos_token='EOS') -> list[str]:
        # 使用CLIP的预处理
        if not torch.is_tensor(img):
            img = self.preprocess(img)
        else:
            # 如果已经是tensor, 由于clip的预处理需要图像, 需要转换回PIL图像再预处理
            transform = transforms.ToPILImage()
            img = self.preprocess(transform(img))

        # 移动到正确设备
        device = next(self.parameters()).device
        img = img.unsqueeze(0).to(device)

        # 使用CLIP提取特征
        img_features = self.clip_model.encode_image(img)

        img_features = img_features.float()

        # 投影特征
        img_features = self.projection(img_features).unsqueeze(0)

        # Beam Search -> 生成描述
        cap_gen = CaptionGenerator(embedder=self.embedder,
                               rnn=self.rnn,
                               classifier=self.classifier,
                               eos_id=self.vocab.index(eos_token),
                               beam_size=beam_size,
                               max_caption_length=max_caption_length,
                               length_normalization_factor=length_normalization_factor)

        sentences, score = cap_gen.beam_search(img_features)
        sentences = [' '.join([self.vocab[idx] for idx in sent])
                 for sent in sentences]

        processed_sentences = []

        for sent in sentences:
            # 去除连续重复的词
            words = sent.split()
            filtered_words = []
            for i, word in enumerate(words):
                if i == 0 or word != words[i-1]:
                    filtered_words.append(word)
            # 去除过多的逗号
            final_sent = " ".join(filtered_words).replace(" , ,", ",").replace(" , ", ", ")
            processed_sentences.append(final_sent)

        return processed_sentences


    def save_checkpoint(self, filepath: str, save_dir: str, keep_count: int, current_epoch: int):
        '''
        保存检查点的权重.

        :param filepath: 完整的检查点文件路径
        :param save_dir: 保存目录路径
        :param keep_count: 滚动保存中保留的数量, 早于当前保存的文件的前keep_count个的文件将被删除, 防止占用磁盘空间过大, 设置为0或负数则不会删除
        :param current_epoch: 当前训练的epoch数
        '''
        import os
        import glob
        import logging

        # 保存当前检查点
        checkpoint_data = {
            'embedder_dict': self.embedder.state_dict(),
            'rnn_dict': self.rnn.state_dict(),
            'projection_dict': self.projection.state_dict(),
            'classifier_dict': self.classifier.state_dict(),
            'vocab': self.vocab,
            'model': self,
        }

        torch.save(checkpoint_data, filepath)
        logging.debug(f"检查点已保存: {filepath}")

        if keep_count > 0:
            # 查找所有检查点文件
            checkpoint_pattern = os.path.join(save_dir, 'checkpoint_epoch_*.pth')
            checkpoint_files = glob.glob(checkpoint_pattern)

            if len(checkpoint_files) > keep_count:
                # 按epoch数字排序 (从文件名中提取epoch数字)
                def extract_epoch(filename):
                    try:
                        basename = os.path.basename(filename)
                        epoch_str = basename.replace('checkpoint_epoch_', '').replace('.pth', '')
                        return int(epoch_str)
                    except:
                        return 0

                checkpoint_files.sort(key=extract_epoch)

                files_to_delete = checkpoint_files[:-keep_count]
                for old_file in files_to_delete:
                    try:
                        os.remove(old_file)
                        logging.debug(f"移除旧的检查点文件: {old_file}")
                    except OSError as e:
                        logging.warning(f"无法移除 {old_file}: {e}")

    def load_checkpoint(self, filename):
        cpnt = torch.load(filename)
        self.embedder.load_state_dict(cpnt['embedder_dict'])
        self.rnn.load_state_dict(cpnt['rnn_dict'])
        if 'projection_dict' in cpnt:
            self.projection.load_state_dict(cpnt['projection_dict'])
        self.classifier.load_state_dict(cpnt['classifier_dict'])

    def finetune_clip(self, optimizer, current_epoch: int, total_epochs: int, base_clip_lr: float, stages: int = 4):
        """
        渐进式解冻CLIP模型，支持多种编码器架构

        参数:
            current_epoch: int - 当前训练的epoch (从0开始)
            total_epochs: int - 总的训练epochs数
            base_clip_lr: float - 基础学习率
            stages: int - 解冻阶段数量
        """
        epoch_per_stage = total_epochs // stages
        current_stage = min((current_epoch // epoch_per_stage) + 1, stages)

        # 默认冻结所有参数
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # 根据模型架构获取对应的transformer blocks
        encoder_name = self.encoder_name

        if encoder_name.startswith('ViT'):
            # Vision Transformer 架构
            encoder_blocks = self.clip_model.visual.transformer.resblocks
        elif encoder_name.startswith('RN'):
            # ResNet 架构
            encoder_blocks = self.clip_model.visual.layer4
        else:
            logging.warning(f"未知的编码器架构: {encoder_name}, 默认使用ViT解冻策略")
            encoder_blocks = self.clip_model.visual.transformer.resblocks

        total_blocks = len(encoder_blocks)

        # 根据当前阶段解冻
        if current_stage == 1:
            # 阶段1: 完全冻结CLIP
            self.clip_model.eval()
        elif current_stage == 2:
            # 阶段2: 解冻最后1/4的解码器块
            blocks_to_unfreeze = max(1, total_blocks // 4)
            for i in range(total_blocks - blocks_to_unfreeze, total_blocks):
                for param in encoder_blocks[i].parameters():
                    param.requires_grad = True
            self.clip_model.train()
        elif current_stage == 3:
            # 阶段3: 解冻后半部分解码器块
            blocks_to_unfreeze = total_blocks // 2
            for i in range(total_blocks - blocks_to_unfreeze, total_blocks):
                for param in encoder_blocks[i].parameters():
                    param.requires_grad = True
            self.clip_model.train()
        else:  # current_stage == 4
            # 阶段4: 解冻全部层
            for param in self.clip_model.parameters():
                param.requires_grad = True
            self.clip_model.train()

        clip_lr = base_clip_lr * (1 - (current_stage / (stages + 1)))     # +1平滑防止clip_lr变成0

        # 更新学习率
        optimizer.param_groups[0]['lr'] = clip_lr

        logging.info(f"Stage {current_stage}/{stages}: CLIP fine-tuning (epoch {current_epoch}/{total_epochs-1})")
        if current_stage == 1:
            logging.info("Stage 1: Fully frozen: no finetuning.")
        elif current_stage == 2:
            logging.info(f"Stage 2: Finetuning: Last 25% encoder layers of {encoder_name}.")
        elif current_stage == 3:
            logging.info(f"Stage 3: Finetuning: Last 50% encoder layers of {encoder_name}.")
        else:
            logging.info(f"Stage >= 4: Finetuning All layers of {encoder_name}.")

        # 在每个stage内部也使用余弦调度器逐渐降低
        # 计算当前stage内的相对进度
        stage_length = total_epochs // 4
        stage_epoch = current_epoch % stage_length
        stage_progress = stage_epoch / stage_length

        current_clip_lr = optimizer.param_groups[0]['lr']
        learning_rate = current_clip_lr

        if stage_progress > 0.5:  # 只在每个stage的后半部分降低学习率
            cosine_factor = 0.5 * (math.cos(2.0 * math.pi * (stage_progress - 0.5)) + 1)   # 0.5 * (cos(2π(x-0.5)) + 1), 在[0,1]上值域[0,1]
            learning_rate = current_clip_lr * cosine_factor
            optimizer.param_groups[0]['lr'] = learning_rate

        if current_stage > 1:
            logging.info(f"Stage{current_stage}/Epoch{current_epoch}: learning rate = {learning_rate}")

        return current_stage
