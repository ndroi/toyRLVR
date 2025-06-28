import numpy as np
import torch

from config import *
from tokenizer import Tokenizer, think, res, eos, numbers, operators, sep


class Actor:
    def __init__(self, model: torch.nn.Module, tokenizer: Tokenizer, mode: str = 'sample', constraint=True):
        self._tokenizer = tokenizer
        self._model = model
        self._model.eval()
        assert mode in ['sample', 'greedy']
        self._mode = mode
        self._constraint = constraint
        self._constraint_think_mask = None
        self._constraint_res_mask = None
        if self._constraint:
            all_tokens = list(range(self._tokenizer.vocab_size()))
            # constraint decoding in think phase
            self._constraint_think_mask = np.zeros_like(all_tokens, dtype=np.float32)
            think_phase_words = numbers + operators + [sep, res]
            constraint_think_tokens = set(self._tokenizer.encode(''.join(think_phase_words)))
            for token in all_tokens:
                if token not in constraint_think_tokens:
                    self._constraint_think_mask[token] = float('-inf')
            # constraint decoding in res phase
            self._constraint_res_mask = np.zeros_like(all_tokens, dtype=np.float32)
            res_phase_words = numbers + [eos]
            constraint_res_tokens = set(self._tokenizer.encode(''.join(res_phase_words)))
            for token in all_tokens:
                if token not in constraint_res_tokens:
                    self._constraint_res_mask[token] = float('-inf')

    def _constraint_decoding(self, pre_text: str, next_token_logit: torch.Tensor) -> torch.Tensor:
        if res not in pre_text:  # think phase
            next_token_logit += self._constraint_think_mask
        else:  # res phase
            constraint_res_mask = self._constraint_res_mask
            if pre_text.endswith(res):
                constraint_res_mask = constraint_res_mask.copy()
                constraint_res_mask[self._tokenizer.encode('-')] = 0  # token '-' is allowed
            next_token_logit += constraint_res_mask
        return next_token_logit

    def run(self, expr: str) -> dict:
        text = expr + think if THINK_MODE else expr + res
        tokens = self._tokenizer.encode(text)
        probs = [1.0] * (len(tokens) - 1)
        past_kvs = None
        while True:
            input_tokens = tokens[-1:] if past_kvs else tokens
            with torch.inference_mode():
                _input = torch.tensor(data=np.array(input_tokens).reshape(1, -1), dtype=torch.int)
                logit, past_kvs = self._model(_input, past_kvs=past_kvs)
                next_token_logit = logit[0][-1]
                if self._constraint:
                    next_token_logit = self._constraint_decoding(text, next_token_logit)
                next_token_prob = torch.softmax(next_token_logit, dim=-1).detach().numpy()
            if self._mode == 'sample':
                next_token = np.random.choice(len(next_token_prob), p=next_token_prob)
                probs.append(next_token_prob[next_token])
            else:
                next_token = np.argmax(next_token_prob)
                probs.append(next_token_prob[next_token])
            next_word = self._tokenizer.decode([next_token])
            tokens.append(next_token)
            text += next_word
            if len(tokens) >= MAX_TOKEN_NUM or next_token == self._tokenizer.eos_token:
                break
        out = {
            'text': text,
            'tokens': tokens,
            'probs': probs,
        }
        return out
