import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 数据预处理
def preprocess_data(file_path, step_size=20):
    # 读取CSV文件
    data = pd.read_csv(file_path, header=None, usecols=range(7))
    data.columns = ['timestamp', 'id', 'open', 'high', 'low', 'close', 'vol']

    # 去掉多余的列
    # data = data.dropna(axis=1)

    # 将时间戳转换为日期格式
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

    # 按币种名称分组
    grouped = data.groupby('id')

    # 提取每个币种的时间序列数据
    crypto_data = {}
    for name, group in grouped:
        # 按时间戳排序
        group = group.sort_values(by='timestamp')
        crypto_data[name] = group[['timestamp', 'open', 'high', 'low', 'close', 'vol']].values

    # 构建训练和测试数据集
    def create_sequences(data, step_size):
        X, y = [], []
        data_len = len(data)

        # 如果数据长度不足，填充数据
        if data_len < 2 * step_size:
            padding = np.tile(data[-1], (2 * step_size - data_len, 1))  # 用最后一个值填充
            data = np.concatenate([data, padding], axis=0)
        for i in range(len(data) - 2 * step_size + 1):  # 确保输入和目标不重叠
            X.append(data[i:i + step_size, 1:])  # 输入序列：从 i 到 i+step_size-1
            y.append(data[i + step_size:i + 2 * step_size, 1:])  # 目标序列：从 i+step_size 到 i+2*step_size-1
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    train_data = []
    test_data = []
    for name, data in crypto_data.items():
        # 将时间戳转换为日期
        timestamps = data[:, 0]

        # 划分训练集和测试集
        train_mask = timestamps < pd.to_datetime('2025-01-01')  # 2024年及以前的数据
        test_mask = timestamps >= pd.to_datetime('2025-01-01')  # 2025年的数据

        train_X, train_y = create_sequences(data[train_mask], step_size)
        test_X, test_y = create_sequences(data[test_mask], step_size)

        train_data.append((train_X, train_y))
        test_data.append((test_X, test_y))

    # 合并所有币种的数据
    train_X = np.concatenate([x for x, y in train_data])
    train_y = np.concatenate([y for x, y in train_data])
    test_X = np.concatenate([x for x, y in test_data])
    test_y = np.concatenate([y for x, y in test_data])

    # 转换为PyTorch张量
    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    test_X = torch.tensor(test_X, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)

    return train_X, train_y, test_X, test_y


def normalize_data(data):
    """
    对数据进行 Z-Score 归一化。
    :param data: 输入数据，形状为 (num_samples, seq_len, input_dim)
    :return: 归一化后的数据
    """
    mean = data.mean(dim=(0, 1), keepdim=True)  # 计算均值和标准差
    std = data.std(dim=(0, 1), keepdim=True)

    # 归一化
    normalized_data = (data - mean) / (std + 1e-8)  # 添加一个小常数避免除零

    return normalized_data, mean, std

def denormalize_data(normalized_data, mean, std):
    """
    将归一化后的数据还原到原始范围。
    :param normalized_data: 归一化后的数据
    :param mean: 归一化时使用的均值
    :param std: 归一化时使用的标准差
    :return: 反归一化后的数据
    """
    return normalized_data * std + mean

# 2. 构建Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()

        # Embedding layers
        self.src_embedding = nn.Linear(input_dim, model_dim)
        self.tgt_embedding = nn.Linear(input_dim, model_dim)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, model_dim))  # 假设序列长度最大为1000

        # Transformer
        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 设置batch_first=True
        )
        # Output layer
        self.fc_out = nn.Linear(model_dim, input_dim)

    def forward(self, src, tgt):
        # src: (batch_size, src_seq_len, input_dim)
        # tgt: (batch_size, tgt_seq_len, input_dim)

        # Embedding and positional encoding
        src_seq_len = src.size(1)
        tgt_seq_len = tgt.size(1)

        src = self.src_embedding(src)  # (batch_size, src_seq_len, model_dim)
        src = src + self.positional_encoding[:, :src_seq_len, :]

        tgt = self.tgt_embedding(tgt)  # (batch_size, tgt_seq_len, model_dim)
        tgt = tgt + self.positional_encoding[:, :tgt_seq_len, :]

        # Transformer
        output = self.transformer(src, tgt)  # (tgt_seq_len, batch_size, model_dim)

        # Final output layer
        output = self.fc_out(output)  # (tgt_seq_len, batch_size, input_dim)

        return output


# 3. 训练模型
def train_model(model, train_X, train_y, epochs=100, batch_size=32, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    dataset = TensorDataset(train_X, train_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            # print(batch_X)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X, batch_X)  # 自回归任务，输入和输出相同
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")


# 4. 测试模型
def test_model(model, test_X, test_y, test_y_mean, test_y_std):
    model.eval()
    with torch.no_grad():
        test_X = test_X.to(device)
        test_y = test_y.to(device)

        predictions = model(test_X, test_X)
        loss = nn.MSELoss()(predictions, test_y)
        print(f"Test Loss: {loss.item():.4f}")

        # 反归一化
        predictions = denormalize_data(predictions, test_y_mean, test_y_std)
        test_y = denormalize_data(test_y, test_y_mean, test_y_std)

        # 可视化预测结果
        plt.figure(figsize=(10, 6))
        plt.plot(test_y[0, :, 0].numpy(), label='True Values')
        plt.plot(predictions[0, :, 0].numpy(), label='Predictions')
        plt.legend()
        plt.show()


# 5. 主函数
def main():
    # 数据预处理
    file_path = 'D:\common\projs\MoneyPrinter\candles.csv'
    train_X, train_y, test_X, test_y = preprocess_data(file_path, step_size=20)

    print(f'\nUsing device: {device}')
    # 数据归一化
    train_X, train_X_mean, train_X_std = normalize_data(train_X)
    train_y, train_y_mean, train_y_std = normalize_data(train_y)
    test_X, test_X_mean, test_X_std = normalize_data(test_X)
    test_y, test_y_mean, test_y_std = normalize_data(test_y)

    # 将数据移动到 GPU
    train_X = train_X.to(device)
    train_y = train_y.to(device)
    test_X = test_X.to(device)
    test_y = test_y.to(device)
    test_y_mean = test_y_mean.to(device)
    test_y_std = test_y_std.to(device)

    # 模型参数
    input_dim = 5  # 开盘价、最高价、最低价、收盘价、数量
    model_dim = 32
    num_heads = 4
    num_layers = 4
    dim_feedforward = 256
    dropout = 0.2

    # 初始化模型
    model = TransformerModel(input_dim, model_dim, num_heads, num_layers, dim_feedforward, dropout)

    # 将模型移动到 GPU
    model = model.to(device)

    # 训练模型
    train_model(model, train_X, train_y, epochs=50, batch_size=1024, lr=0.001)

    # 测试模型
    test_model(model, test_X, test_y, test_y_mean, test_y_std)


# 运行主函数
if __name__ == '__main__':
    main()