import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import math
import matplotlib.pyplot as plt


class CryptoDataset(Dataset):
    def __init__(self, data, sequence_length=20):
        self.sequence_length = sequence_length
        self.normalized_data = []

        # 预处理所有序列进行归一化
        for i in range(len(data) - sequence_length):
            full_sequence = data[i:i + sequence_length + 1]
            scaler = MinMaxScaler()
            norm_sequence = scaler.fit_transform(full_sequence)
            self.normalized_data.append(norm_sequence)

        self.normalized_data = np.array(self.normalized_data)

    def __len__(self):
        return len(self.normalized_data)

    def __getitem__(self, idx):
        sequence = self.normalized_data[idx][:-1]  # 前20个
        target = self.normalized_data[idx][-1]  # 第21个
        return torch.FloatTensor(sequence), torch.FloatTensor(target)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model, dropout=dropout,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, src):
        embedded = self.embedding(src)
        embedded = self.pos_encoder(embedded)
        memory = self.transformer_encoder(embedded)
        return memory


class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=4 * d_model, dropout=dropout,
                                                    batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, tgt, memory):
        output = self.transformer_decoder(tgt, memory)
        output = self.output_layer(output)
        return output


class CryptoTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=128, nhead=8, num_encoder_layers=4, num_decoder_layers=4):
        super(CryptoTransformer, self).__init__()
        self.encoder = TransformerEncoder(input_dim, d_model, nhead, num_encoder_layers)
        self.decoder = TransformerDecoder(output_dim, d_model, nhead, num_decoder_layers)
        self.tgt_embedding = nn.Linear(input_dim, d_model)

    def forward(self, src, tgt):
        memory = self.encoder(src)
        tgt = self.tgt_embedding(tgt)
        output = self.decoder(tgt, memory)
        return output


class LossHistory:
    def __init__(self):
        self.train_losses = []

    def add_train_loss(self, loss):
        self.train_losses.append(loss)

    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot( self.train_losses, label='Training Loss')
        plt.xticks(range(len(self.train_losses)))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()


def prepare_data(df, sequence_length=20, split_date='2025-01-01'):
    """
    处理数据，获取训练集和测试集
    :param df: 原始数据dataframe
    :param sequence_length: 序列长度
    :param split_date: 划分日期
    :return:
    """
    # 检查是否已经保存了归一化后的数据
    if os.path.exists('../data/train_normalized.npy') and os.path.exists('../data/test_normalized.npy'):
        train_data = np.load('../data/train_normalized.npy')
        test_data = np.load('../data/test_normalized.npy')
        return LoadedDataset(train_data), LoadedDataset(test_data)
    # 划分训练集和测试集
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    train_df = df[df['timestamp'] < split_date]
    test_df = df[df['timestamp'] >= split_date]

    features = ['open', 'high', 'low', 'close', 'vol']
    train_dataset = CryptoDataset(train_df[features].values, sequence_length)
    test_dataset = CryptoDataset(test_df[features].values, sequence_length)

    os.makedirs('../data', exist_ok=True)
    np.save('../data/train_normalized.npy', train_dataset.normalized_data)
    np.save('../data/test_normalized.npy', test_dataset.normalized_data)

    return train_dataset, test_dataset


class LoadedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx][:-1]), torch.FloatTensor(self.data[idx][-1])

    def __len__(self):
        return len(self.data)


def train(model, train_loader, criterion, optimizer, device, num_epochs=20):
    """
    训练模型
    :param model: 模型
    :param train_loader: 训练数据加载器
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param device: 设备（cuda或cpu）
    :param num_epochs: 训练轮数
    :return:
    """
    history = LossHistory()

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data, data)
            loss = criterion(output[:, -1, :], target)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        history.add_train_loss(avg_train_loss)
        print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}')

    return history


def test(model, test_loader, criterion, device):
    """
    测试模型
    :param model: 模型
    :param test_loader: 测试数据加载器
    :param criterion: 损失函数
    :param device: 设备（cuda或cpu）
    :return:
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, data)
            loss = criterion(output[:, -1, :], target)
            total_loss += loss.item()
    return total_loss / len(test_loader)


def evaluate_predictions(model, test_loader, device, num_sequences=100):
    """
    评估模型预测结果并绘制比较图
    :param model: 模型
    :param test_loader: 测试数据加载器
    :param device: 设备（cuda或cpu）
    :param num_sequences: 选取的序列数量
    :return:
    """
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            if i >= num_sequences:
                break
            data = data.to(device)
            output = model(data, data)
            predictions.append(output[:, -1, :].cpu().numpy())
            targets.append(target.numpy())

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    # 绘制比较图
    plt.figure(figsize=(15, 5))
    features = ['open', 'high', 'low', 'close', 'vol']

    for i, feature in enumerate(features):
        plt.subplot(1, 5, i + 1)
        plt.scatter(targets[:, i], predictions[:, i], alpha=0.5)
        plt.plot([targets[:, i].min(), targets[:, i].max()],
                 [targets[:, i].min(), targets[:, i].max()], 'r--')
        plt.title(feature)
        plt.xlabel('True')
        plt.ylabel('Predicted')

    plt.tight_layout()
    plt.show()

    # 计算metrics
    mae = np.mean(np.abs(predictions - targets))
    mse = np.mean((predictions - targets) ** 2)
    print(f'MAE: {mae:.4f}, MSE: {mse:.4f}')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv('D:\common\projs\MoneyPrinter\candles.csv', header=None, usecols=range(7))
    df.columns = ['timestamp', 'id', 'open', 'high', 'low', 'close', 'vol']

    train_dataset, test_dataset = prepare_data(df, split_date='2025-01-01')
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    input_dim = 5  # open, high, low, close, volume
    output_dim = 5
    model = CryptoTransformer(input_dim, output_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("开始训练...")
    history = train(model, train_loader, criterion, optimizer, device)

    print("\n开始测试...")
    test_loss = test(model, test_loader, criterion, device)
    print(f'最终测试损失: {test_loss:.4f}')
    history.plot_losses()
    torch.save(model.state_dict(), '../models/crypto_transformer.pth')
    # model.load_state_dict(torch.load('../models/crypto_transformer.pth'))
    # evaluate_predictions(model, test_loader, device)


if __name__ == "__main__":
    main()