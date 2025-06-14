import os

import torch
import torch.multiprocessing as mp
from torch import nn
from torch import optim

from buffer import Buffer
from config import *
from model import Model
from tokenizer import Tokenizer

device = 'cuda:0'


class RLTrainer:
    def __init__(self, buffer: Buffer, model_lock: mp.Lock):
        self._buffer = buffer
        self._model_lock = model_lock
        self._model = Model(vocab_size=Tokenizer().vocab_size())
        self._model.to(device)
        self._init_ckpt()
        self._optimizer = optim.Adam(self._model.parameters(), lr=LEARNING_RATE)

    def _init_ckpt(self):
        if os.path.exists(CKPT_PATH):
            self._model.load_state_dict(torch.load(CKPT_PATH))
            print(f'[RL-Trainer] model ckpt is loaded from {CKPT_PATH}.')
        else:
            self._save_ckpt()
            print(f'[RL-Trainer] an empty model ckpt is created in {CKPT_PATH}.')

    def _save_ckpt(self):
        assert self._model
        with self._model_lock:
            torch.save(self._model.state_dict(), CKPT_PATH)

    def _run_step(self, step: int):
        sample = self._buffer.batch(TRAINING_BATCH_SIZE)
        tokens = torch.tensor(data=sample['tokens'], dtype=torch.int, device=device)
        _return = torch.tensor(data=sample['reward'], dtype=torch.float32, device=device).view(-1, 1)
        mask = torch.tensor(data=sample['loss_mask'], dtype=torch.bool, device=device)
        old_action_prob = torch.tensor(data=sample['probs'], dtype=torch.float32, device=device)
        advantage = torch.tensor(data=sample['advantages'], dtype=torch.float32, device=device)
        state = tokens[:, :-1]
        action = tokens[:, 1:]

        logit, value, _ = self._model(state)

        # policy loss
        prob = nn.functional.softmax(logit, dim=-1)
        action_prob = torch.gather(prob, dim=-1, index=action.unsqueeze(-1).long()).squeeze()
        rate = action_prob / old_action_prob
        policy_obj = torch.minimum(advantage * rate, advantage * torch.clip(rate, 0.8, 1.2))
        masked_policy_obj = policy_obj[mask]
        policy_loss = -torch.mean(masked_policy_obj)

        # value loss
        _value_loss = 0.5 * torch.square(value - _return)
        masked_value_loss = _value_loss[mask]
        value_loss = torch.mean(masked_value_loss)

        # entropy loss
        masked_prob = prob[mask]
        # entropy = -torch.sum(masked_prob * torch.log(masked_prob), dim=-1)
        # standard entropy is week for this task.
        entropy = -torch.sum(torch.log(masked_prob), dim=-1)
        entropy_loss = torch.mean(entropy)

        # total loss
        loss = policy_loss + value_loss + entropy_loss * 1e-4

        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()
        if (step + 1) % 1000 == 0:
            self._save_ckpt()
        if (step + 1) % 10 == 0:
            print('-' * 30)
            print(f'[RL-Trainer] step: {step} text: {sample["text"][0]}')
            print(
                f'[RL-Trainer] '
                f'step: {step} '
                f'reward: {sample["reward"].mean():.4f} '
            )
            print(
                f'[RL-Trainer] '
                f'step: {step} '
                f'policy-loss: {policy_loss:.4f} '
                f'value-loss: {value_loss:.4f} '
                f'entropy-loss: {entropy_loss:.4f} '
                f'total-loss: {loss:.4f}',
            )

    def run(self):
        for step in range(TRAINING_STEP):
            self._run_step(step)
        self._save_ckpt()
