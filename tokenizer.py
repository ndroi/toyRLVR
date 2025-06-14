from typing import List

numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
operators = ['+', '-', '*']
think = '<THINK>'
res = '<RES>'
eos = '<EOS>'
special = [think, res, eos]
extra = ['a', 'b', 'c', 'd']


class Tokenizer:
    def __init__(self):
        self._vocab = numbers + operators + special + extra
        self._word2id = {}
        self._word2id = {word: i for i, word in enumerate(self._vocab)}
        self._id2word = {i: word for i, word in enumerate(self._vocab)}
        self.think_token = self._word2id[think]
        self.res_token = self._word2id[res]
        self.eos_token = self._word2id[eos]

    def vocab_size(self) -> int:
        return len(self._vocab)

    def encode(self, text: str) -> List[int]:
        id_list = []
        while text:
            spec = False
            for s in special:
                if text.startswith(s):
                    id_list.append(self._word2id[s])
                    text = text[len(s):]
                    spec = True
                    break
            if not spec:
                id_list.append(self._word2id[text[0]])
                text = text[1:]
        return id_list

    def decode(self, id_list: List[int]) -> str:
        return ''.join([self._id2word[_id] for _id in id_list])
