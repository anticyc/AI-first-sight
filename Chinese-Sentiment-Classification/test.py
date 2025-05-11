import os
from data_loader import load_data
from models import CNN, LSTM, MLP, LSTMBi
from train import DATA_ROOT_DIR, word2vec_path
from tqdm import tqdm
import numpy as np
import torch
import argparse


def test(model, test_loader):
    criterion = torch.nn.BCEWithLogitsLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        correct = 0
        total = 0
        TP, TN, FP, FN = 0, 0, 0, 0
        epoch_val_losses = []
        # 使用 tqdm 显示进度条
        for vectors, ori_labels in tqdm(test_loader, desc="Testing"):
            vectors = vectors.float().to(device)
            labels = (
                torch.nn.functional.one_hot(ori_labels, num_classes=2)
                .float()
                .to(device)
            )
            ori_labels = ori_labels.to(device)
            outputs = model(vectors)
            loss = criterion(outputs, labels)
            epoch_val_losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == ori_labels).sum().item()
            TP += ((ori_labels == 1) & (predicted == 1)).sum().item()
            TN += ((ori_labels == 0) & (predicted == 0)).sum().item()
            FP += ((ori_labels == 1) & (predicted == 0)).sum().item()
            FN += ((ori_labels == 0) & (predicted == 1)).sum().item()
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        accuracy = correct / total
        print(f"Test Loss: {np.mean(epoch_val_losses):.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")


def model_resolver(model: str, path: str):
    model = model.lower()
    if model == "cnn":
        _model_ = CNN()
    elif model == "rnn":
        _model_ = LSTM()
    elif model == "rnn2":
        _model_ = LSTMBi()
    elif model == "mlp":
        _model_ = MLP()
    else:
        raise ValueError("Invalid model name")
    _model_.load_state_dict(torch.load(path))
    test_loader = load_data(os.path.join(DATA_ROOT_DIR, "test.txt"), word2vec_path)
    test(_model_, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cnn", help="model name")
    parser.add_argument("--path", type=str, default="model.pth", help="path to model")
    args = parser.parse_args()
    model_resolver(args.model, args.path)
