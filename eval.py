import torch

from actor import Actor
from config import *
from expr import gen_expr
from model import Model
from tokenizer import Tokenizer

EVAL_COUNT = 1


def main():
    tokenizer = Tokenizer()
    model = Model(vocab_size=tokenizer.vocab_size())
    model.load_state_dict(torch.load(CKPT_PATH))
    actor = Actor(model=model, tokenizer=tokenizer, mode='greedy')
    for i in range(EVAL_COUNT):
        expr, result = gen_expr()
        out = actor.run(expr)
        # print(out)
        text = out['text']
        print(f'[Eval] eval-{i}, expr: {expr} true-result: {result} text: {text}')


if __name__ == '__main__':
    main()
