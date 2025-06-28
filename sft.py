import os
import random

import numpy as np
import torch
from torch import nn
from torch import optim

from config import *
from expr import gen_expr
from model import Model
from tokenizer import Tokenizer, think, res, eos, operators, sep

device = 'cuda:0'


class SFTDataset:
    def __init__(self):
        self._tokenizer = Tokenizer()
        self._response_token = self._tokenizer.think_token if THINK_MODE else self._tokenizer.res_token

    def _thinking(self, expr: str) -> str:
        def reduce(_expr: str) -> str:
            _numbers = []
            _tmp_num = ''
            for i, x in enumerate(_expr):
                if x in operators and i > 0:
                    _numbers.append(int(_tmp_num))
                    _tmp_num = x
                else:
                    _tmp_num += x
                    if i == len(_expr) - 1:
                        _numbers.append(int(_tmp_num))
            if len(_numbers) <= 1:
                return _expr
            k = random.randint(0, len(_numbers) - 2)
            _numbers[k] = _numbers[k] + _numbers[k + 1]
            _numbers.pop(k + 1)
            _reduced = ''
            for i, _num in enumerate(_numbers):
                if _num > 0 and i > 0:
                    _reduced += '+' + str(_num)
                else:
                    _reduced += str(_num)
            return _reduced

        thinking = ''
        while True:
            reduced = reduce(expr)
            if reduced == expr:
                break
            expr = reduced
            thinking += reduced + sep
        thinking = thinking.strip(sep)
        return thinking

    def generate(self) -> (np.ndarray, np.ndarray):
        expr, result = gen_expr()
        text = expr
        if THINK_MODE:
            text += think + self._thinking(expr)
        text += res + str(result) + eos
        tokens = self._tokenizer.encode(text)
        input_ids = np.array(tokens[:-1])
        labels = np.array(tokens[1:])
        response_idx = tokens.index(self._response_token)
        labels[:response_idx] = -100
        return input_ids, labels

    def batch(self, batch_size: int) -> dict:
        assert batch_size >= 1
        samples = [self.generate() for _ in range(batch_size)]
        max_len = max([len(s[0]) for s in samples])
        batch = {
            'input_ids': np.zeros((batch_size, max_len), dtype=np.int32),
            'labels': np.zeros((batch_size, max_len), dtype=np.int32) - 100,
        }
        for i, sample in enumerate(samples):
            _len = len(sample[0])
            batch['input_ids'][i, :_len] = sample[0]
            batch['labels'][i, :_len] = sample[1]
        return batch


class SFTTrainer:
    def __init__(self, dataset: SFTDataset):
        self._model = Model(vocab_size=Tokenizer().vocab_size())
        self._model.to(device)
        self._init_ckpt()
        self._optimizer = optim.Adam(self._model.parameters(), lr=LEARNING_RATE)
        self._dataset = dataset
        self._ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def _init_ckpt(self):
        if os.path.exists(CKPT_PATH):
            self._model.load_state_dict(torch.load(CKPT_PATH))
            print(f'[SFT-Trainer] model ckpt is loaded from {CKPT_PATH}.')
        else:
            self._save_ckpt()
            print(f'[SFT-Trainer] an empty model ckpt is created in {CKPT_PATH}.')

    def _save_ckpt(self):
        assert self._model
        torch.save(self._model.state_dict(), CKPT_PATH)

    def _run_step(self, step: int):
        sample = self._dataset.batch(TRAINING_BATCH_SIZE)
        input_ids = torch.tensor(data=sample['input_ids'], dtype=torch.int, device=device)
        labels = torch.tensor(data=sample['labels'], dtype=torch.float32, device=device)
        logit, _ = self._model(input_ids)
        logit = logit.view(-1, logit.size(2))
        labels = labels.view(-1)
        loss = self._ce_loss(logit, labels.to(torch.int64))
        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()
        if (step + 1) % 1000 == 0:
            self._save_ckpt()
        if (step + 1) % 10 == 0:
            print(f'[SFT-Trainer] step: {step} loss: {loss:.4f}')

    def run(self):
        for step in range(TRAINING_STEP):
            self._run_step(step)
        self._save_ckpt()


def main():
    dataset = SFTDataset()
    trainer = SFTTrainer(dataset=dataset)
    trainer.run()


if __name__ == '__main__':
    main()
