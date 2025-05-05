import json
import math
import operator
import sys

pinyindict = {}

EPSILON = 1e-8
TRI_RATIO = 0.63


class Node:
    def __init__(self, pinyin, word, weight=-30, pred=None):
        self.pinyin = pinyin
        self.word = word
        self.weight = weight
        self.pred = pred


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
    with open("data/temp/3_word.txt", "r", encoding="gbk") as f:
        three_word_data = json.load(f)
        for key, value in three_word_data.items():
            first_word = key[:-2]
            three_word_data[key] = (
                -30
                if value == 0.0
                else math.log(float(value) / two_word_data[first_word])
            )
    for key, value in two_word_data.items():
        first_word = key.split()[0]
        two_word_data[key] = (
            -30 if value == 0.0 else math.log(float(value) / one_word_data[first_word])
        )
    return one_word_data, two_word_data, three_word_data


def viterbi(query_list, one_word_data, two_word_data, three_word_data):
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
                if i >= 2:
                    pred_2 = pred.pred
                    connect = pred_2.word + " " + pred.word + " " + node.word
                    tri_weight = three_word_data.get(connect, -30)
                    bi_weight = two_word_data.get(connect[2:], -30)
                    new_weight = (
                        TRI_RATIO * tri_weight + (1 - TRI_RATIO) * bi_weight
                    ) 
                    if new_weight + pred.weight > best_weight:
                        best_weight = new_weight + pred.weight
                        node.pred = pred
                else:
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
    one_word_data, two_word_data, three_word_data = train()
    while True:
        try:
            line = sys.stdin.readline()
            if line is None or line == "":
                break
            query_list = line.strip().split()
            result = viterbi(query_list, one_word_data, two_word_data, three_word_data)
            print(result)
        except EOFError:
            break


def generate(lines: list):
    import time
    print("Loading data...")
    start = time.time()
    one_word_data, two_word_data, three_word_data = train()
    results = []
    train_time = time.time() - start
    print("Generating results...")
    for line in lines:
        query_list = line.strip().split()
        result = viterbi(query_list, one_word_data, two_word_data, three_word_data)
        results.append(result)
    run_time = time.time() - start - train_time
    return results, train_time, run_time


if __name__ == "__main__":
    main()
