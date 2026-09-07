import os
from rich.console import Console
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import string
from pycocotools.coco import COCO
from random import randrange
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from preprocessing import Sharpen

COCO_IMG_PATH = "RSTPReid_new/"
COCO_ANN_PATH = "RSTPReid_new/annotations/"

# COCO_IMG_PATH = "coco2014/"
# COCO_ANN_PATH = "coco2014/annotations/"

TRAIN_PATH = {'root': os.path.join(COCO_IMG_PATH, 'train2014'),
                'annFile': os.path.join(COCO_ANN_PATH, 'captions_train2014.json')
                }
VAL_PATH = {'root': os.path.join(COCO_IMG_PATH, 'val2014'),
              'annFile': os.path.join(COCO_ANN_PATH, 'captions_val2014.json')
              }

UNK_TOKEN = 'UNK'
PAD_TOKEN = 'PAD'
EOS_TOKEN = 'EOS'

normalize = {'mean': [0.485, 0.456, 0.406],
               'std': [0.229, 0.224, 0.225]}

console = Console()


def simple_tokenize(captions):
    processed = []
    for j, s in enumerate(captions):
        s = str(s).lower()
        for punct in string.punctuation:
            s = s.replace(punct, f' {punct} ')
        txt = s.strip().split()
        processed.append(txt)
    return processed

def build_vocab(annFile=TRAIN_PATH['annFile'], num_words=10000):
    # count up the number of words
    counts = {}
    coco = COCO(annFile)
    ids = coco.imgs.keys()
    for img_id in ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        captions = simple_tokenize([ann['caption'] for ann in anns])
        for txt in captions:
            for w in txt:
                counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)

    vocab = [w for (_, w) in cw[:num_words]]
    vocab = [PAD_TOKEN] + vocab + [UNK_TOKEN, EOS_TOKEN]

    return vocab

class CaptionProcessor:
    def __init__(self, vocab, rnd_caption=True):
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.unk = self.word2idx[UNK_TOKEN]
        self.rnd_caption = rnd_caption

    def __call__(self, captions):
        captions = simple_tokenize(captions)
        if self.rnd_caption:
            idx = randrange(len(captions))
        else:
            idx = 0
        caption = captions[idx]
        targets = []
        for w in caption:
            targets.append(self.word2idx.get(w, self.unk))
        return torch.Tensor(targets)


class BatchCollator:
    def __init__(self, vocab, max_length=50):
        self.padding = vocab.index(PAD_TOKEN)
        self.eos = vocab.index(EOS_TOKEN)
        self.max_length = max_length

    def __call__(self, img_cap):
        img_cap.sort(key=lambda p: len(p[1]), reverse=True)
        imgs, caps = zip(*img_cap)
        imgs = torch.cat([img.unsqueeze(0) for img in imgs], 0)
        lengths = [min(len(c) + 1, self.max_length) for c in caps]
        batch_length = max(lengths)

        cap_tensor = torch.LongTensor(len(caps), batch_length).fill_(self.padding)

        for i, c in enumerate(caps):
            end_cap = lengths[i] - 1
            if end_cap < batch_length:
                cap_tensor[i, end_cap] = self.eos
            cap_tensor[i, :end_cap].copy_(c[:end_cap])

        return (imgs, (cap_tensor, lengths))


def create_target(vocab, rnd_caption=True):
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    unk = word2idx[UNK_TOKEN]

    def get_caption(captions):
        captions = simple_tokenize(captions)
        if rnd_caption:
            idx = randrange(len(captions))
        else:
            idx = 0
        caption = captions[idx]
        targets = []
        for w in caption:
            targets.append(word2idx.get(w, unk))
        return torch.Tensor(targets)
    return get_caption

def create_batches(vocab, max_length=50):
    padding = vocab.index(PAD_TOKEN)
    eos = vocab.index(EOS_TOKEN)

    def collate(img_cap):
        img_cap.sort(key=lambda p: len(p[1]), reverse=True)
        imgs, caps = zip(*img_cap)
        imgs = torch.cat([img.unsqueeze(0) for img in imgs], 0)
        lengths = [min(len(c) + 1, max_length) for c in caps]
        batch_length = max(lengths)

        # cap_tensor的形状应该是[batch_size, batch_length], 不是[batch_length, batch_size]
        cap_tensor = torch.LongTensor(len(caps), batch_length).fill_(padding)

        for i, c in enumerate(caps):
            end_cap = lengths[i] - 1
            if end_cap < batch_length:
                cap_tensor[i, end_cap] = eos

            # 索引顺序相应变化: [i, :end_cap], 不是[:end_cap, i]
            cap_tensor[i, :end_cap].copy_(c[:end_cap])

        return (imgs, (cap_tensor, lengths))
    return collate

def get_coco_data(vocab, train=True, img_size=224, scale_size=256, normalize=normalize, target_img_size=None):
    if train:
        root, annFile = TRAIN_PATH['root'], TRAIN_PATH['annFile']
        train_transforms = []
        if target_img_size: # 如果指定了特定尺寸，则使用
            train_transforms.append(transforms.Resize(target_img_size))
            # 对于Re-ID模型，不需要RandomCrop，因为Resize已经固定了尺寸
        else: # 否则使用常规的缩放和裁剪
            train_transforms.append(transforms.Resize(scale_size))
            train_transforms.append(transforms.RandomCrop(img_size))

        train_transforms.extend([
            transforms.RandomApply([Sharpen(factor=1.5)], p=0.5), # 条件性添加
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize(**normalize)
        ])

        img_transform = transforms.Compose(train_transforms)

    else:
        root, annFile = VAL_PATH['root'], VAL_PATH['annFile']
        val_transforms = []
        if target_img_size:
            val_transforms.append(transforms.Resize(target_img_size))
        else:
            val_transforms.append(transforms.Resize(scale_size))
            val_transforms.append(transforms.CenterCrop(img_size))
        val_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(**normalize)
        ])
        img_transform = transforms.Compose(val_transforms)

    caption_processor = CaptionProcessor(vocab, rnd_caption=train)

    data = (dset.CocoCaptions(root=root, annFile=annFile, transform=img_transform,
                              target_transform=caption_processor), vocab)

    return data

def get_iterator(data, batch_size=32, max_length=30, shuffle=True, num_workers=4, pin_memory=True):
    cap, vocab = data

    cap0 = [vocab[int(word)] for word in cap[0][1]]
    console.log(f"数据集第0个caption: cap0: {cap0}", style="bold blue")

    batch_collator = BatchCollator(vocab, max_length=max_length)

    return torch.utils.data.DataLoader(
        cap,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=batch_collator,  # 使用类的实例
        num_workers=num_workers,              # 现在可以在 Windows 上正常工作了
        pin_memory=pin_memory
    )
