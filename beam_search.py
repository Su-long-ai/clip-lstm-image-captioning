"""Beam-search decoding utilities for the CLIP + LSTM caption model.

The structure is adapted from TensorFlow's historical im2txt caption generator,
but the PyTorch execution path is modernized and works on both CPU and CUDA.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import torch
from torch.nn.functional import log_softmax


@dataclass
class Caption:
    sentence: list[int]
    state: object
    logprob: float
    score: float
    metadata: object | None = None

    def __lt__(self, other: "Caption") -> bool:
        return self.score < other.score


class TopN:
    """Maintain the top-N scored elements from an incremental stream."""

    def __init__(self, n: int):
        if n <= 0:
            raise ValueError("n must be positive")
        self._n = n
        self._data: list | None = []

    def size(self) -> int:
        if self._data is None:
            raise RuntimeError("TopN was extracted; call reset() before reuse")
        return len(self._data)

    def push(self, item) -> None:
        if self._data is None:
            raise RuntimeError("TopN was extracted; call reset() before reuse")
        if len(self._data) < self._n:
            heapq.heappush(self._data, item)
        else:
            heapq.heappushpop(self._data, item)

    def extract(self, *, sort: bool = False) -> list:
        if self._data is None:
            raise RuntimeError("TopN was already extracted")
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self) -> None:
        self._data = []


class CaptionGenerator:
    def __init__(
        self,
        embedder,
        rnn,
        classifier,
        eos_id: int,
        beam_size: int = 3,
        max_caption_length: int = 40,
        length_normalization_factor: float = 0.0,
    ):
        if beam_size <= 0:
            raise ValueError("beam_size must be positive")
        if max_caption_length <= 0:
            raise ValueError("max_caption_length must be positive")
        self.embedder = embedder
        self.rnn = rnn
        self.classifier = classifier
        self.eos_id = int(eos_id)
        self.beam_size = int(beam_size)
        self.max_caption_length = int(max_caption_length)
        self.length_normalization_factor = float(length_normalization_factor)

    def _topk_words(self, embeddings: torch.Tensor, state):
        with torch.no_grad():
            output, new_states = self.rnn(embeddings, state)
            logits = self.classifier(output.squeeze(0))
            logprobs = log_softmax(logits, dim=1)
            k = min(self.beam_size, logprobs.shape[1])
            values, words = logprobs.topk(k, dim=1)
        return words, values, new_states

    @staticmethod
    def _merge_states(captions: list[Caption]):
        states = [caption.state for caption in captions]
        first = states[0]
        if isinstance(first, tuple):
            hs, cs = zip(*states, strict=True)
            return torch.cat(hs, dim=1), torch.cat(cs, dim=1)
        return torch.cat(states, dim=1)

    @staticmethod
    def _slice_state(states, index: int):
        if isinstance(states, tuple):
            return states[0].narrow(1, index, 1), states[1].narrow(1, index, 1)
        return states.narrow(1, index, 1)

    def beam_search(self, rnn_input: torch.Tensor, initial_state=None) -> tuple[list[list[int]], list[float]]:
        """Generate captions for one image feature sequence.

        `rnn_input` must have shape `[1, 1, embedding_dim]`. All intermediate
        token tensors are created on the same device, so CPU and CUDA follow the
        same decoding path.
        """

        if rnn_input.ndim != 3 or rnn_input.shape[0] != 1 or rnn_input.shape[1] != 1:
            raise ValueError("rnn_input must have shape [1, 1, embedding_dim]")

        partial = TopN(self.beam_size)
        complete = TopN(self.beam_size)
        words, logprobs, new_state = self._topk_words(rnn_input, initial_state)

        for k in range(words.shape[1]):
            word = int(words[0, k].item())
            logprob = float(logprobs[0, k].item())
            caption = Caption([word], new_state, logprob, logprob)
            if word == self.eos_id:
                complete.push(caption)
            else:
                partial.push(caption)

        for _ in range(self.max_caption_length - 1):
            if partial.size() == 0:
                break
            partial_list = partial.extract()
            partial.reset()

            input_feed = torch.tensor(
                [caption.sentence[-1] for caption in partial_list],
                dtype=torch.long,
                device=rnn_input.device,
            )
            state_feed = self._merge_states(partial_list)
            with torch.no_grad():
                embeddings = self.embedder(input_feed).view(1, len(input_feed), -1)
                words, logprobs, new_states = self._topk_words(embeddings, state_feed)

            for i, partial_caption in enumerate(partial_list):
                state = self._slice_state(new_states, i)
                for k in range(words.shape[1]):
                    word = int(words[i, k].item())
                    word_logprob = float(logprobs[i, k].item())
                    sentence = partial_caption.sentence + [word]
                    logprob = partial_caption.logprob + word_logprob
                    score = logprob
                    if word == self.eos_id and self.length_normalization_factor > 0:
                        score /= len(sentence) ** self.length_normalization_factor
                    beam = Caption(sentence, state, logprob, score)
                    if word == self.eos_id:
                        complete.push(beam)
                    else:
                        partial.push(beam)

        if complete.size() == 0:
            complete = partial
        captions = complete.extract(sort=True)
        return [caption.sentence for caption in captions], [caption.score for caption in captions]

