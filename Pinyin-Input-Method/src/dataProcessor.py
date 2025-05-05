# data processor for the input method
import os
import json
import ast
import argparse
import pypinyin
import re


def parser():
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="corpus/sina_news_gbk")
    parser.add_argument("--output", type=str, default="data/temp")
    parser.add_argument("--pinyin", type=bool, default=False)
    return parser.parse_args()


class DataProcessor:
    def __init__(self, folder):
        self.folder = folder
        self.one_wd = {}
        self.two_wd = {}
        self.three_wd = {}
        self.one_py = {}
        with open("data/一二级汉字表.txt", "r", encoding="gbk") as f:
            words = f.read()
            for word in words:
                self.one_wd[word] = 0

    def traverse(self, document):
        last = None
        last_2 = None
        for char in document:
            if char in self.one_wd:
                self.one_wd[char] += 1
                if last is not None:
                    self.two_wd[last + " " + char] = (
                        self.two_wd.get(last + " " + char, 0) + 1
                    )
                    if last_2 is not None:
                        self.three_wd[last_2 + " " + last + " " + char] = (
                            self.three_wd.get(last_2 + " " + last + " " + char, 0) + 1
                        )
                last_2 = last
                last = char
            else:
                last = last_2 = None

    def do(self, pinyin):
        for file in os.listdir(self.folder):
            if file == "README.txt":
                continue
            if not file.endswith(".txt"):
                continue
            with open(os.path.join(self.folder, file), "r", encoding="gbk") as f:
                lines = f.readlines()
                for line in lines:
                    json_brick = ast.literal_eval(line)
                    # process the json brick
                    if pinyin:
                        self.pytraverse(json_brick["title"])
                        self.pytraverse(json_brick["html"])
                    else:
                        self.traverse(json_brick["title"])
                        self.traverse(json_brick["html"])

    def output(self, output_folder, pinyin):
        if not pinyin:
            with open(os.path.join(output_folder, "2_word.txt"), "w") as f:
                json.dump(self.two_wd, f, ensure_ascii=False)
            with open(os.path.join(output_folder, "3_word.txt"), "w") as f:
                json.dump(
                    self.three_wd, f, ensure_ascii=False
                )  # 三字情形只在无拼音情况下考察
            with open(os.path.join(output_folder, "1_word.txt"), "w") as f:
                json.dump(self.one_wd, f, ensure_ascii=False)  # 不含有拼音的字典
        else:
            with open(os.path.join(output_folder, "py_2_word.txt"), "w") as f:
                json.dump(self.two_wd, f, ensure_ascii=False)
            with open(os.path.join(output_folder, "py_1_word.txt"), "w") as f:
                json.dump(self.one_py, f, ensure_ascii=False)  # 含有拼音的字典

    def pytraverse(self, document):
        # 利用正则表达式，把非中文换成空格
        document = re.sub(r"[^\u4e00-\u9fa5]", " ", document).strip()
        # replace chars in documents that are not in the one_wd dict
        document = "".join([c if c in self.one_wd else " " for c in document])
        pieces = document.split()
        for p in pieces:
            if not p:  # 跳过空字符串
                continue
            try:
                py = pypinyin.lazy_pinyin(p)
                if len(p) != len(py):  # 长度不匹配情况则跳过
                    continue
                for ch, pinyin in zip(p, py):
                    if ch == " ":
                        continue
                    if pinyin == "lve":
                        pinyin = "lue"
                    if pinyin in self.one_py:
                        self.one_py[pinyin][ch] = self.one_py[pinyin].get(ch, 0) + 1
                    else:
                        self.one_py[pinyin] = {ch: 1}
                for i in range(len(py) - 1):
                    if p[i] == " " or p[i + 1] == " ":
                        continue
                    dual = p[i] + " " + p[i + 1]
                    dual_py = (
                        (py[i] if py[i] != "lve" else "lue")
                        + " "
                        + (py[i + 1] if py[i + 1] != "lve" else "lue")
                    )
                    if dual_py in self.two_wd:
                        if dual in self.two_wd[dual_py]:
                            self.two_wd[py[i] + " " + py[i + 1]][dual] += 1
                        else:
                            self.two_wd[py[i] + " " + py[i + 1]][dual] = 1
                    else:
                        self.two_wd[py[i] + " " + py[i + 1]] = {dual: 1}
            except Exception as e:  # 处理拼音转换可能出现的异常
                print(f"Error processing {p}: {e}")
                continue


if __name__ == "__main__":
    args = parser()
    dp = DataProcessor(args.folder)
    dp.do(args.pinyin)
    dp.output(args.output, args.pinyin)
