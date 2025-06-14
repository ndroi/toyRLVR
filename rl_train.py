import os

import torch.multiprocessing as mp

from config import *


def rollout_worker(sample_queue: mp.Queue, model_lock: mp.Lock):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    from rollout import Rollout
    rollout = Rollout(sample_queue=sample_queue, model_lock=model_lock)
    rollout.run()


def run_trainer(sample_queue: mp.Queue, model_lock: mp.Lock):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import torch
    torch.cuda.set_device('cuda:0')
    from buffer import Buffer
    from rl_trainer import RLTrainer
    buffer = Buffer(sample_queue)
    trainer = RLTrainer(buffer=buffer, model_lock=model_lock)
    trainer.run()


def main():
    mp.set_start_method('spawn')
    sample_queue = mp.Queue(maxsize=32)
    model_lock = mp.Lock()
    rollout_processes = [
        mp.Process(target=rollout_worker, args=(sample_queue, model_lock)) for _ in range(N_ROLLOUT_PROCESS)
    ]
    for p in rollout_processes:
        p.daemon = True
        p.start()
    run_trainer(sample_queue, model_lock)


if __name__ == '__main__':
    main()
