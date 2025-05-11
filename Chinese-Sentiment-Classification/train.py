import torch
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch.optim.lr_scheduler import ExponentialLR
from datetime import datetime
from data_loader import load_data
from models import CNN, LSTM, MLP, LSTMBi

DATA_ROOT_DIR = "./Dataset"
word2vec_path = os.path.join(DATA_ROOT_DIR, "wiki_word2vec_50.bin")

def train_model(_model_: str, _epoch_: str, _msg_: str):
    # 以下代码为随机选择一个GPU以均衡负载
    # import random
    # cuda_no = random.randint(0, 7)
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_no)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, criterion, optimizer = None, None, None
    _model_ = _model_.lower()
    if _model_ == "cnn":
        model = CNN().to(device)
    elif _model_ == "rnn":
        model = LSTM().to(device)
    elif _model_ == "rnn2":
        model = LSTMBi().to(device)
    elif _model_ == "mlp":
        model = MLP().to(device)
    else:
        raise ValueError("Invalid model type")
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr, weight_decay=1e-6)
    scheduler = ExponentialLR(optimizer, gamma=model.exp)
    train_loader = load_data(os.path.join(DATA_ROOT_DIR, "train.txt"), word2vec_path)
    validation_loader = load_data(
        os.path.join(DATA_ROOT_DIR, "validation.txt"), word2vec_path
    )
    num_epochs = int(_epoch_)
    losses, val_loss = [], []
    f1_vec, precision_vec, recall_vec, accu, accuracy_vec = [], [], [], [], []
    for epoch in range(num_epochs):
        correct, total = 0, 0
        model.train()
        epoch_losses = []
        for vectors, ori_labels in tqdm(train_loader):
            vectors = vectors.float().to(device)
            labels = (
                torch.nn.functional.one_hot(ori_labels, num_classes=2)
                .float()
                .to(device)
            )
            optimizer.zero_grad()
            outputs = model(vectors)
            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == ori_labels.to(device)).sum().item()
                total += labels.size(0)
            loss = criterion(outputs, labels)
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        scheduler.step()
        losses.append(np.mean(epoch_losses))
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        accuracy = correct / total
        accu.append(accuracy)
        print(f"Training Accuracy: {accuracy:.4f}")

        with torch.no_grad():
            correct = 0
            total = 0
            TP, TN, FP, FN = 0, 0, 0, 0
            model.eval()
            epoch_val_losses = []
            # 使用 tqdm 显示进度条
            for vectors, ori_labels in tqdm(validation_loader):
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
            val_loss.append(np.mean(epoch_val_losses))
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            precision_vec.append(precision)
            recall_vec.append(recall)
            f1 = 2 * precision * recall / (precision + recall)
            f1_vec.append(f1)
            accuracy = correct / total
            accuracy_vec.append(accuracy)
            print(f"Validation Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
    plt.plot(losses, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend(["Train Loss", "Validation Loss"])
    plt.grid()
    plt.title(f"Loss Curve of {_model_} Model")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(_model_ + "_" + _msg_ + "_loss.png")
    plt.clf()

    plt.title(f"Validation Details of {_model_} Model")
    plt.plot(precision_vec, label="Precision")
    plt.plot(recall_vec, label="Recall")
    plt.plot(f1_vec, label="F1 Score")
    plt.plot(accuracy_vec, label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend(["Precision", "Recall", "F1 Score", "Accuracy"])
    plt.grid()
    plt.savefig(_model_ + "_" + _msg_ + "_val.png")
    plt.clf()

    plt.plot(accu, label="Train Accuracy")
    plt.plot(accuracy_vec, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Train Accuracy", "Validation Accuracy"])
    plt.grid()
    plt.title(f"Accuracy Curve of {_model_} Model")
    plt.savefig(_model_ + "_" + _msg_ + "_accu.png")
    plt.clf()

    torch.save(model.state_dict(), _model_ + "_" + _msg_ + "_model.pth")


if __name__ == "__main__":
    now = datetime.now()  # 时间戳
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cnn", help="model type")
    parser.add_argument("--epoch", type=int, default=500, help="epoch nums")
    parser.add_argument(
        "--message", type=str, default=now.strftime("%d-%H:%M"), help="message"
    )
    args = parser.parse_args()
    train_model(args.model, args.epoch, args.message)
