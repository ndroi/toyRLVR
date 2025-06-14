import numpy as np
import torch

from actor import Actor
from expr import gen_expr
from reward import calc_reward
from tokenizer import Tokenizer


class Rollout:
    def __init__(self, model: torch.nn.Module, tokenizer: Tokenizer):
        self._actor = Actor(model=model, tokenizer=tokenizer)
        self._tokenizer = tokenizer

    def generate(self) -> dict:
        expr, result = gen_expr()
        text, tokens = self._actor.run(expr)
        think_token_idx = tokens.index(self._tokenizer.think_token)
        reward = calc_reward(result, text[think_token_idx + 1:])
        response_mask = np.zeros_like(tokens, dtype=np.float32)
        response_mask[think_token_idx + 1:] = 1
        sample = {
            'text': text,
            'tokens': np.array(tokens, dtype=np.int32),
            'response_begin': response_mask,
            'reward': reward,
            'return': reward,
        }
        return sample
