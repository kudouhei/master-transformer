
import re
import random
from nltk.tokenize import word_tokenize
from collections import defaultdict, deque


class MarkovChain:
    def __init__(self):
        self.lookup_dict = defaultdict(list)
        self._seeded = False
        self.__seed_me()

    def __seed_me(self, rand_seed=None):
        if self._seeded is not True:
            try:
                if rand_seed is not None:
                    random.seed(rand_seed)
                else:
                    random.seed()
                self._seeded = True
            except NotImplementedError:
                self._seeded = False

    def add_document(self, str):
        preprocessed_list = self._preprocess(str)
        pairs = self.__generate_tuple_keys(preprocessed_list)
        for pair in pairs:
            self.lookup_dict[pair[0]].append(pair[1])

    def _preprocess(self, str):
        cleaned = re.sub(r"\W+", " ", str).lower()  # 移除非字母数字字符并转为小写
        tokenized = word_tokenize(cleaned)  # 使用NLTK分词
        return tokenized

    def __generate_tuple_keys(self, data):
        if len(data) < 1:
            return

        for i in range(len(data) - 1):
            yield [data[i], data[i + 1]]

    def generate_text(self, max_length=50):
        context = deque()  # 双端队列，用于存储上下文
        output = []
        if len(self.lookup_dict) > 0:
            self.__seed_me(rand_seed=len(self.lookup_dict))
            chain_head = [list(self.lookup_dict)[0]]  # 从字典中随机选择一个词作为初始词
            context.extend(chain_head)

            while len(output) < (max_length - 1):
                next_choices = self.lookup_dict[context[-1]]  # 获取当前词的下一个词
                if len(next_choices) > 0:
                    next_word = random.choice(next_choices)  # 随机选择一个下一个词
                    context.append(next_word)  # 将下一个词添加到上下文
                    output.append(context.popleft())  # 将上下文中的第一个词添加到输出
                else:
                    break
            output.extend(list(context))
        return " ".join(output)


if __name__ == "__main__":
    with open("./data/hamlet.txt", "r", encoding="utf-8") as f:
        text = f.read()
        # print(text)
    HMM = MarkovChain()
    HMM.add_document(text)

    print(HMM.generate_text(max_length=25))
