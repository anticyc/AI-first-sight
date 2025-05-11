import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.lr = 6e-5
        self.exp = 0.995
        self.hidden_size = 80
        self.kernel_sizes = [2, 3, 4]
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(50, self.hidden_size, kernel_size=k),
                    nn.BatchNorm1d(self.hidden_size),
                    nn.GELU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                for k in self.kernel_sizes
            ]
            + [
                nn.Sequential(
                    nn.Conv1d(50, self.hidden_size, kernel_size=k),
                    nn.BatchNorm1d(self.hidden_size),
                    nn.GELU(),
                    nn.AdaptiveMaxPool1d(1),
                )
                for k in self.kernel_sizes
            ]
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(len(self.kernel_sizes) * 2 * self.hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch_size, 50, max_len] -> [batch_size, max_len, 50]
        convs = [conv(x) for conv in self.convs]
        convs = torch.cat(convs, dim=1)
        convs = convs.view(convs.size(0), -1)
        return self.fc(convs)


class LSTM(nn.Module):
    def __init__(self):
        self.hidden_size = 32
        self.lr = 3e-5
        self.exp = 1.0
        self.bidirectional = False
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=50,
            hidden_size=self.hidden_size,
            num_layers=2,
            dropout=0.7,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        self.fc = nn.Linear((2 if self.bidirectional else 1) * self.hidden_size, 2)
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[(n // 4) : (n // 2)].fill_(1)
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步的输出
        out = self.fc(out)  # 全连接层
        return out

class LSTMBi(nn.Module):
    def __init__(self):
        super(LSTMBi, self).__init__()
        self.hidden_size = 32
        self.lr = 3e-5
        self.exp = 1.0
        self.bidirectional = True
        self.lstm = nn.LSTM(
            input_size=50,
            hidden_size=self.hidden_size,
            num_layers=2,
            dropout=0.7,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        self.bn = nn.BatchNorm1d(self.hidden_size * 2)
        self.fc = nn.Linear((2 if self.bidirectional else 1) * self.hidden_size, 2)
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[(n // 4) : (n // 2)].fill_(1)
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步的输出
        out = self.bn(out)  # 批量归一化层
        out = self.fc(out)  # 全连接层
        return out

class MLP(nn.Module):
    def __init__(self, max_len: int = 70):
        super(MLP, self).__init__()
        self.lr = 4e-6
        self.exp = 0.99
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(max_len * 50, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        input = x.view(x.size(0), -1)
        return self.fc(input)
