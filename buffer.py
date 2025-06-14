import random
import time
from queue import Empty
from threading import Thread

import numpy as np
import torch.multiprocessing as mp


class Buffer:
    def __init__(self, sample_queue: mp.Queue, max_buffer_size=2048):
        self._sample_queue = sample_queue
        self._max_buffer_size = max_buffer_size
        self._samples = []
        Thread(target=self._background_worker).start()

    def _background_worker(self):
        while True:
            try:
                sample = self._sample_queue.get(block=False)
                self._samples.append(sample)
                del sample
                if len(self._samples) > self._max_buffer_size:
                    del self._samples[0]
            except Empty:
                time.sleep(0.1)

    def batch(self, batch_size: int) -> dict:
        assert 1 <= batch_size <= self._max_buffer_size
        min_samples_count = min(16 * batch_size, self._max_buffer_size)
        while len(self._samples) < min_samples_count:
            print(f'[Buffer] waiting for samples({len(self._samples)}/{min_samples_count})...')
            time.sleep(1.0)
        samples = random.sample(self._samples, batch_size)
        # padding to the same length.
        max_tokens_len = max([len(s['tokens']) for s in samples])
        batch = {
            'text': [''] * batch_size,
            'tokens': np.zeros((batch_size, max_tokens_len), dtype=np.int32),
            'loss_mask': np.zeros((batch_size, max_tokens_len - 1), dtype=np.int32),
            'reward': np.zeros(batch_size, dtype=np.float32),
            'probs': np.ones((batch_size, max_tokens_len - 1), dtype=np.float32),
            'advantages': np.zeros((batch_size, max_tokens_len - 1), dtype=np.float32),
        }
        for i, sample in enumerate(samples):
            _len = len(sample['tokens'])
            batch['text'][i] = sample['text']
            batch['tokens'][i, :_len] = sample['tokens']
            batch['loss_mask'][i, :_len - 1] = sample['loss_mask']
            batch['reward'][i] = sample['reward']
            batch['probs'][i, :_len - 1] = sample['probs']
            batch['advantages'][i, :_len - 1] = sample['advantages']
        return batch
