from tokenizer import res, eos


def calc_reward(true_result: int, response_text: str) -> float:
    # response_text should be like:
    # think-mode: "3+4<THINK>1234abcd<RES>7<EOS>"
    # non-think-mode： "3+4<RES>7<EOS>"
    if not response_text.endswith(eos):
        return 0
    if res not in response_text:
        return 0
    res_idx = response_text.find(res)
    # think_text = response_text[:res_idx]
    result_text = response_text[res_idx + len(res):-len(eos)]
    try:
        result = int(result_text)
    except ValueError:
        return 0
    if result != true_result:
        return 0
    return 1
