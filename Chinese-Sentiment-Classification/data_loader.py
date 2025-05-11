import gensim
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


noise_std = 0.03

def load_data(
    data_path: str, word2vec_path: str, batch_size: int = 256, max_len: int = 70
):
    # binary word2vec loader
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(
        word2vec_path, binary=True
    )
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        vectors, signs = [], []
        for line in lines:
            words = line.strip().split()
            line_vec = []
            signs.append(int(words[0]))
            for word in words[1:]:
                noise = np.random.normal(0, noise_std, 50)
                line_vec.append(
                    word2vec[word] + noise if word in word2vec else np.random.normal(0, 1, 50)
                )
            line_vec = np.stack(line_vec)
            # padding according to max length
            if len(line_vec) < max_len:
                padding = np.zeros((max_len - len(line_vec), 50))  # padding matrix
                line_vec = np.vstack((line_vec, padding))  # 将填充矩阵堆叠到原向量后面
            else:
                line_vec = line_vec[: max_len - 1]  # 截断到最大长度
                line_vec = np.vstack((line_vec, np.zeros((1, 50))))  # 添加eos向量
            vectors.append(line_vec)
        dataset = TensorDataset(
            torch.tensor(np.stack(vectors)), torch.tensor(np.stack(signs))
        )
    # returns a dataloader
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
    )
