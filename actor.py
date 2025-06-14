from typing import List

import numpy as np
import torch

from common import *
from tokenizer import Tokenizer, think


class Actor:
    def __init__(self, model: torch.nn.Module, tokenizer: Tokenizer):
        self._tokenizer = tokenizer
        self._model = model

    def run(self, expr: str) -> (str, List[int]):
        text = expr + think
        tokens = self._tokenizer.encode(text)
        past_kvs = None
        while True:
            input_tokens = tokens[-1:] if past_kvs else tokens
            with torch.inference_mode():
                _input = torch.tensor(data=np.array(input_tokens).reshape(1, -1), dtype=torch.int)
                logit, past_kvs = self._model(_input, past_kvs=past_kvs)
                next_token_logit = logit[0][-1]
                next_token_prob = torch.softmax(next_token_logit, dim=0).detach().numpy()
            next_token = np.random.choice(len(next_token_prob), p=next_token_prob)
            next_word = self._tokenizer.decode([next_token])
            tokens.append(next_token)
            text += next_word
            if len(tokens) >= MAX_TOKEN_NUM or next_token == self._tokenizer.eos_token:
                break
        return text, tokens
