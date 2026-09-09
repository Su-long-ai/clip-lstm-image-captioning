from __future__ import annotations

import torch

from data import (
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    BatchCollator,
    CaptionProcessor,
    build_vocab_from_captions,
    simple_tokenize,
)


def test_simple_tokenize_separates_punctuation() -> None:
    assert simple_tokenize(["Hello, World!"]) == [["hello", ",", "world", "!"]]


def test_vocab_is_frequency_then_lexical_and_has_special_tokens() -> None:
    vocab = build_vocab_from_captions(["beta alpha", "beta gamma", "alpha delta"], num_words=3)
    assert vocab == [PAD_TOKEN, "alpha", "beta", "delta", UNK_TOKEN, EOS_TOKEN]


def test_caption_processor_returns_long_and_unknown_token() -> None:
    vocab = [PAD_TOKEN, "known", UNK_TOKEN, EOS_TOKEN]
    result = CaptionProcessor(vocab, rnd_caption=False)(["known missing"])
    assert result.dtype == torch.long
    assert result.tolist() == [1, 2]


def test_batch_collator_appends_eos_and_truncates() -> None:
    vocab = [PAD_TOKEN, "a", "b", "c", UNK_TOKEN, EOS_TOKEN]
    collate = BatchCollator(vocab, max_length=3)
    image_a = torch.zeros(3, 2, 2)
    image_b = torch.ones(3, 2, 2)
    captions = [torch.tensor([1, 2, 3], dtype=torch.long), torch.tensor([1], dtype=torch.long)]
    images, (tokens, lengths) = collate([(image_a, captions[0]), (image_b, captions[1])])
    assert images.shape == (2, 3, 2, 2)
    assert lengths == [3, 2]
    assert tokens.tolist() == [[1, 2, 5], [1, 5, 0]]
