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
        while True:
            self._update_model_param()
            samples = self.generate(group_size=ROLLOUT_GROUP_SIZE)
            for sample in samples:
                self._sample_queue.put(sample)

    def generate(self, group_size: int) -> [dict]:
        expr, result = gen_expr()
        out_group = []
        for _ in range(group_size):
            out = self._actor.run(expr)
            text = out['text']
            reward = calc_reward(result, text)
            tokens = out['tokens']
            response_idx = tokens.index(self._response_token)
            loss_mask = np.zeros_like(tokens[:-1], dtype=np.float32)
            loss_mask[response_idx:] = 1
            out['loss_mask'] = loss_mask
            out['reward'] = reward
            out_group.append(out)
        reward_group = [out['reward'] for out in out_group]
        mean = np.mean(reward_group)
        std = np.std(reward_group) + 1e-8
        for out in out_group:
            out['advantage'] = (out['reward'] - mean) / std
        return out_group
