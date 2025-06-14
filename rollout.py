import os

import numpy as np
import torch
import torch.multiprocessing as mp

from actor import Actor
from config import *
from expr import gen_expr
from model import Model
from reward import calc_reward
from tokenizer import Tokenizer


class Rollout:
    def __init__(self, sample_queue: mp.Queue, model_lock: mp.Lock):
        self._parent_pid = os.getppid()
        self._sample_queue = sample_queue
        self._model_lock = model_lock
        self._tokenizer = Tokenizer()
        self._model = Model(vocab_size=self._tokenizer.vocab_size())
        self._model_param_time = -1
        self._actor = Actor(model=self._model, tokenizer=self._tokenizer)
        self._response_token = self._tokenizer.think_token if THINK_MODE else self._tokenizer.res_token

    def _update_model_param(self):
        model_param_time = os.stat(CKPT_PATH).st_mtime if os.path.exists(CKPT_PATH) else 0
        if self._model_param_time >= model_param_time:
            return
        with self._model_lock:
            print('[Rollout] update model ckpt from trainer...')
            self._model.load_state_dict(torch.load(CKPT_PATH, map_location=torch.device('cpu')))
        self._model_param_time = model_param_time if model_param_time > 0 else os.stat(CKPT_PATH).st_mtime

    def run(self):
        update_model_step = 16
        while True:
            self._update_model_param()
            for _ in range(update_model_step):
                sample = self.generate()
                self._sample_queue.put(sample)

    def generate(self) -> dict:
        expr, result = gen_expr()
        out = self._actor.run(expr)
        text = out['text']
        tokens = out['tokens']
        response_idx = tokens.index(self._response_token)
        reward = calc_reward(result, text)
        loss_mask = np.zeros_like(tokens[:-1], dtype=np.float32)
        loss_mask[response_idx:] = 1
        out['loss_mask'] = loss_mask
        out['reward'] = reward
        out['advantages'] = reward - np.array(out['values'], dtype=np.float32)
        return out
