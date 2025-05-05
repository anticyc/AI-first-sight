import bigram
import trigram
import withpy
import phrase
import numpy as np
import argparse
import tri_phrase


def parser():
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--input", type=str, default="data/input.txt")
    parser.add_argument("--answer", type=str, default="data/answer.txt")
    parser.add_argument("--output", type=str, default="data/output.txt")
    return parser.parse_args()


def generate(args):
    lines = open(args.input, encoding="utf-8").readlines()
    results = []
    with open(args.output, "w", encoding="utf-8") as f:
        if args.model == "bigram":
            results, train_time, run_time = bigram.generate(lines=lines)
        elif args.model == "trigram":
            results, train_time, run_time = trigram.generate(lines=lines)
        elif args.model == "pinyin":
            results, train_time, run_time = withpy.generate(lines=lines)
        elif args.model == "phrase":
            results, train_time, run_time = phrase.generate(lines=lines)
        elif args.model == "triphrase":
            results, train_time, run_time = tri_phrase.generate(lines=lines)
        else:
            print("Error in evaluate.py - 模型参数错误")
            exit(1)
        # write to file
        for line in results:
            f.write(line + "\n")
    print("训练时间: ", round(train_time, 3))
    print("运行时间: ", round(run_time, 3))
    evaluate(results, "../data/answer.txt")


def evaluate(output: list, ans_path: str):
    ans = []
    with open(ans_path, encoding="utf-8") as f:
        ans = f.readlines()
    total, right, right_line = 0, 0, 0
    # 以句为单位评测
    for i in range(len(output)):
        if output[i].strip() == ans[i].strip():
            right_line += 1
        # 以字为单位评测
        x = np.array(list(output[i].strip()))
        y = np.array(list(ans[i].strip()))
        total += x.shape[0]
        try:
            right += np.sum(x == y)
        except ValueError as e:
            # 若句长度不一致，则抛出异常
            print("Error in evaluate.py - 长度不一致，第" + str(i) + "行")
            raise
    print("句子正确率: ", format(right_line / len(output), ".3f"))
    print("单字正确率: ", format(right / total, ".3f"))


if __name__ == "__main__":
    args = parser()
    print("Current Model: " + args.model)
    generate(args)
