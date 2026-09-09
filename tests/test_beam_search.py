from __future__ import annotations

import torch
import torch.nn as nn

from beam_search import Caption, CaptionGenerator, TopN


def test_topn_keeps_highest_scores() -> None:
    top = TopN(2)
    top.push(Caption([1], None, 1.0, 1.0))
    top.push(Caption([2], None, 3.0, 3.0))
    top.push(Caption([3], None, 2.0, 2.0))
    assert [caption.score for caption in top.extract(sort=True)] == [3.0, 2.0]


def test_beam_search_runs_on_cpu_and_can_finish_with_eos() -> None:
    embedding_dim = 4
    hidden_size = 4
    vocab_size = 3
    eos_id = 2

    embedder = nn.Embedding(vocab_size, embedding_dim)
    rnn = nn.LSTM(embedding_dim, hidden_size)
    classifier = nn.Linear(hidden_size, vocab_size)
    with torch.no_grad():
        for parameter in rnn.parameters():
            parameter.zero_()
        embedder.weight.zero_()
        classifier.weight.zero_()
        classifier.bias.copy_(torch.tensor([0.0, -1.0, 5.0]))

    generator = CaptionGenerator(
        embedder,
        rnn,
        classifier,
        eos_id=eos_id,
        beam_size=2,
        max_caption_length=4,
    )
    sentences, scores = generator.beam_search(torch.zeros(1, 1, embedding_dim))
    assert sentences
    assert sentences[0][-1] == eos_id
    assert all(isinstance(token, int) for token in sentences[0])
    assert all(isinstance(score, float) for score in scores)
