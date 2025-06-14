from tokenizer import think, res, eos


def calc_reward(true_result: int, response_text: str) -> float:
    # response_text should be like "1234abcd<res>7<eos>"
    r = 0.0
    if not response_text.endswith(eos):
        return r
    r += 0.1
    if think in response_text:
        return r
    if res in response_text:
        r += 0.1
    result_text = response_text[response_text.find(res) + len(res):-len(eos)]
    try:
        result = int(result_text)
    except ValueError:
        return r
    r += 0.1
    if result == true_result:
        r += 1.0
    return r
