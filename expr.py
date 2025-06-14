import random

from common import *
from tokenizer import operators


def gen_expr() -> (str, int):
    expr = ''
    num_cnt = random.randint(2, MAX_EXPR_NUM_CNT)
    for i in range(num_cnt):
        num = random.randint(0, MAX_EXPR_NUM)
        expr += str(num)
        if i != num_cnt - 1:
            op = random.choice(operators)
            expr += op
    result = eval(expr)
    return expr, result
