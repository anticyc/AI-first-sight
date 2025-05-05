import jieba
import pypinyin
import json
import math
import operator
import sys

pinyindict = {}

EPSILON = 1e-8


class Node:
    def __init__(self, pinyin, word, weight=-30, pred=None):
        self.pinyin = pinyin
        self.word = word
        self.weight = weight
        self.pred = pred


class LineCutter:
    def __init__(self):
        self.min_len = 3  # 词组最小长度
        self.max_len = 8  # 词组最大长度
        self.freq_base_min = 16  # 词组长度为min_len时，录入字典的基准词频
        self.freq_base_larger = 2  # 词组长度大于min_len时，录入字典的基准词频
        self.dict = self.create_phrase_dict()

    def create_phrase_dict(self):
        jieba.initialize()
        pinyins = {}
        for word, freq in jieba.dt.FREQ.items():
            # 只处理词组中全部是中文的情形
            if all("\u4e00" <= char <= "\u9fff" for char in word):
                if (
                    len(word) == self.min_len
                    and len(word) <= self.max_len
                    and freq >= self.freq_base_min
                    or len(word) > self.min_len
                    and freq >= self.freq_base_larger
                ):
                    # 将汉字转换为拼音序列
                    pinyin_list = pypinyin.pinyin(word, style=pypinyin.Style.NORMAL)
                    pinyin_sequence = " ".join([item[0] for item in pinyin_list])
                    # 将拼音序列和频率添加到字典中
                    if pinyin_sequence not in pinyins:
                        pinyins[pinyin_sequence] = word
                    elif jieba.dt.FREQ[pinyins[pinyin_sequence]] < freq:
                        pinyins[pinyin_sequence] = word
        print("Phrase dict created ... ")
        return pinyins

    def cut(self, pinyin_list: list):
        if len(pinyin_list) < self.min_len:
            return pinyin_list, "", None
        # 遍历句子，在字典中查找首个匹配词组
        for i in range(len(pinyin_list) - self.min_len + 1):
            for j in range(self.min_len, self.max_len + 1):
                if " ".join(pinyin_list[i : i + j]) in self.dict:
                    return (
                        pinyin_list[:i],
                        self.dict[" ".join(pinyin_list[i : i + j])],
                        pinyin_list[i + j :],
                    )
        return pinyin_list, "", None


def train():
    one_word_data = None
    two_word_data = None
    # 读入拼音汉字表，统计每个拼音对应的汉字
    with open("data/拼音汉字表.txt", "r", encoding="gbk") as f:
        for line in f:
            word_py = line.strip().split()
            pinyindict[word_py[0]] = [word for word in word_py[1:]]

    with open("data/temp/1_word.txt", "r", encoding="gbk") as f:
        one_word_data = json.load(f)
        # 得到的是word+出现频次的字典，频率对数的计算放在维特比算法中进行。

    with open("data/temp/2_word.txt", "r", encoding="gbk") as f:
        two_word_data = json.load(f)
        for key, value in two_word_data.items():
            # print(key, value)
            first_word = key.split()[0]
            two_word_data[key] = math.log(float(value) / one_word_data[first_word])
    return one_word_data, two_word_data


def viterbi(query_list, one_word_data, two_word_data):
    if len(query_list) == 0:
        return ""
    layers = []
    # Viberti Algorithm
    for i in range(len(query_list)):
        total_count = sum(
            [one_word_data.get(wd, 0) for wd in pinyindict[query_list[i]]]
        )
        layers.append(
            [
                Node(
                    pinyin=query_list[i],
                    word=wd,
                    weight=(
                        -math.inf
                        if one_word_data.get(wd, 0) == 0
                        else math.log(float(one_word_data.get(wd, 0)) / total_count)
                    ),
                )
                for wd in pinyindict[query_list[i]]
            ]
        )  # 词的列表转化为节点的列表

    for i in range(1, len(layers)):
        for node in layers[i]:
            best_weight = -math.inf
            for pred in layers[i - 1]:
                connect = pred.word + " " + node.word
                new_weight = two_word_data.get(connect, -30)
                if new_weight + pred.weight > best_weight:
                    best_weight = new_weight + pred.weight
                    node.pred = pred
            node.weight = best_weight * (1 - EPSILON) + node.weight * EPSILON

    max_node = max(layers[-1], key=operator.attrgetter("weight"))
    result = max_node.word
    while max_node.pred is not None:
        result = max_node.pred.word + result
        max_node = max_node.pred
    return result


def main():
    Linecutter = LineCutter()
    one_word_data, two_word_data = train()
    while True:
        try:
            line = sys.stdin.readline()
            if line is None or line == "":
                break
            query_list = line.strip().split()
            pt1, pt2, pt3 = Linecutter.cut(query_list)
            result = viterbi(pt1, one_word_data, two_word_data) + pt2
            while pt3 is not None:
                pt1, pt2, pt3 = Linecutter.cut(pt3)
                result += viterbi(pt1, one_word_data, two_word_data) + pt2
            print(result)
        except EOFError:
            break


def generate(lines: list):
    import time
    print("Loading data...")
    start = time.time()
    Linecutter = LineCutter()
    one_word_data, two_word_data = train()
    results = []
    train_time = time.time() - start
    print("Generating results...")
    for line in lines:
        query_list = line.strip().split()
        pt1, pt2, pt3 = Linecutter.cut(query_list)
        result = viterbi(pt1, one_word_data, two_word_data) + pt2
        while pt3 is not None:
            pt1, pt2, pt3 = Linecutter.cut(pt3)
            result += viterbi(pt1, one_word_data, two_word_data) + pt2
        results.append(result)
    run_time = time.time() - start - train_time
    return results, train_time, run_time


if __name__ == "__main__":
    main()
