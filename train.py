import os

import torch
from torch import nn
from torch import optim

from common import *
from model import Model
from rollout import Rollout
from tokenizer import Tokenizer


class Trainer:
    def __init__(self):
        self._tokenizer = Tokenizer()
        self._model = Model(MAX_TOKEN_NUM, self._tokenizer.vocab_size())
        self._rollout = Rollout(model=self._model, tokenizer=self._tokenizer)

    def _init_ckpt(self):
        if os.path.exists(CKPT_PATH):
            self._model.load_state_dict(torch.load(CKPT_PATH, weights_only=True))
            print(f'[Trainer] model ckpt is loaded from {CKPT_PATH}.')
        else:
            self._save_ckpt()
            print(f'[Trainer] an empty model ckpt is created in {CKPT_PATH}.')

    def _save_ckpt(self):
        assert self._model
        torch.save(self._model.state_dict(), CKPT_PATH)

    def run(self):
        optimizer = optim.Adam(self._model.parameters(), lr=LEARNING_RATE)

        for step in range(10000):
            sample = self._rollout.generate()
            tokens = torch.tensor(data=sample['tokens'], dtype=torch.int)
            _return = torch.tensor(data=sample['return'], dtype=torch.float32)
            state = tokens[:-1]
            action = tokens[1:]
            logit, _ = self._model(state.unsqueeze(0))
            prob = nn.functional.softmax(logit[0], dim=-1)
            action_prob = torch.gather(prob, dim=-1, index=action.unsqueeze(-1).long()).squeeze()
            loss = -torch.mean(_return * torch.log(action_prob))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (step + 1) % 100 == 0:
                self._save_ckpt()
            print(f'step: {step} text: {sample["text"]} reward: {sample["reward"]}')
            print(f'step: {step} loss: {loss:.4f}')


if __name__ == '__main__':
    trainer = Trainer()
    trainer.run()
