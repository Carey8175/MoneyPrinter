import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

from system_code.core.config import Config
from system_code.core.modules.position import HoldingGroup
from system_code.core.modules.print_system import PrintSystem, logger, BBRsiMaOpenPositionModule


class ClusterTrain(PrintSystem):
    def __init__(self, bar, train_begin_date, train_end_date, inst_id):
        super(ClusterTrain, self).__init__(
            bar=bar,
            begin=train_begin_date,
            end=train_end_date,
            inst_id=inst_id
        )
        self.open_module = BBRsiMaOpenPositionModule()

    def run(self, drop_columns: list = None, n_clusters: int = 2, save: bool = True):
        logger.info(f"[Training] {str(self)}")

        data = self.fetch_data()
        holding_groups = self.open_module.open_position(data)
        kmeans = self.train_cluster(holding_groups[0], n_clusters=n_clusters, save=save, drop_columns=drop_columns)

        logger.info(f"Cluster model has been saved to {Config().MODEL_DIR}")

    def train_cluster(self, group: HoldingGroup, n_clusters: int = 2, save: bool = True, drop_columns: list = None):
        """
        Clustering the holding group
        Args:
            group: 将这个group进行聚类
            n_clusters: 聚类的数量
            save: 是否保存聚类结果
            drop_columns: 聚类时需要删除的列

        Returns:

        """
        if drop_columns is None:
            drop_columns = []

        # 确保需要删除的所有特征列都在 data 中；若没有 volume，可自行去掉或替换
        available_features = [col for col in group[0].signal_candle_info.columns if col not in drop_columns]

        # 构建一个列表，用于存放聚类所需的特征行
        x_list = []
        begin_date = []
        for hp in group:
            # 提取该 signal bar 的特征值
            begin_date.append(hp.begin)
            row: pd.DataFrame = hp.signal_candle_info.iloc[0]
            x_list.append(row[available_features].values.tolist())

        X = np.array(x_list, dtype=float)

        # 数据清洗（如缺失值填充）与缩放（可视需要选择）
        # 这里简单做个 fillna(0) + 标准化示例
        X = np.nan_to_num(X, nan=0.0)
        # scaler = StandardScaler()
       #  X_scaled = scaler.fit_transform(X)

        # 4. 打乱顺序, 但是保持X_scaled和begin_date的对应关系
        np.random.seed(42)
        shuffle_index = np.random.permutation(X.shape[0])
        X = X[shuffle_index]
        begin_date = list(np.array(begin_date)[shuffle_index])

        # 4.2. 聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)

        # # 大概查看一下分类效果
        # labels_d = []
        # for d, l in zip(begin_date, labels):
        #     labels_d.append((d, l))
        # labels = [l for _, l in sorted(labels_d)]

        # 4.3. 聚类模型分析
        df_clusters, cluster_centers, df_clusters_original = self.analyze_clusters(X, kmeans, available_features, X)
        self.visualize_clusters(df_clusters_original, cluster_centers, available_features, n_clusters)

        # save cluster model
        if save:
            model_path = Config().MODEL_DIR
            joblib.dump(kmeans, f"{model_path}/kmeans_{self.inst_id}_{n_clusters}.pkl")

        return kmeans

    def analyze_clusters(self, X_scaled, kmeans, available_features, X_original):
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        n_clusters = kmeans.n_clusters
        df_clusters = pd.DataFrame(X_scaled, columns=available_features)
        df_clusters_original = pd.DataFrame(X_original, columns=available_features)
        df_clusters['Cluster'] = labels
        df_clusters_original['Cluster'] = labels

        # 1. 统计每个簇的样本数量
        cluster_counts = df_clusters['Cluster'].value_counts().sort_index()
        print("各簇样本数量:")
        print(cluster_counts)

        # 2. 计算每个簇的特征均值和方差
        cluster_summary = df_clusters_original.groupby("Cluster").agg(['mean', 'std'])
        print("\n簇内特征统计 (均值 & 标准差):")
        print(cluster_summary.to_string())

        # 3. 计算轮廓系数, 越接近 1 表示聚类效果越好
        silhouette_avg = silhouette_score(X_scaled, labels)
        print(f"\n轮廓系数: {silhouette_avg:.4f}")

        return df_clusters, cluster_centers, df_clusters_original

    def visualize_clusters(self, df_clusters_o, cluster_centers, available_features, n_clusters):
        # 1. 选取部分特征，绘制箱线图
        top_features = available_features[:10]
        df_melted = df_clusters_o.melt(id_vars=['Cluster'], value_vars=top_features, var_name='Feature',
                                     value_name='Value')
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Feature', y='Value', hue='Cluster', data=df_melted)
        plt.xticks(rotation=45)
        plt.title("不同簇中前 10 个特征的分布")
        plt.show()

        # 2. 雷达图分析聚类中心
        n_radars = 12
        selected_features = available_features[:n_radars]  # 选择 6 个特征
        cluster_radar = pd.DataFrame(cluster_centers[:, :n_radars], columns=selected_features)

        angles = np.linspace(0, 2 * np.pi, len(selected_features), endpoint=False).tolist()
        cluster_radar = pd.concat([cluster_radar, cluster_radar.iloc[:, 0]], axis=1)  # 使雷达图闭合
        angles.append(angles[0])

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'polar': True})
        for i in range(n_clusters):
            ax.plot(angles, cluster_radar.iloc[i], label=f'Cluster {i}')
            ax.fill(angles, cluster_radar.iloc[i], alpha=0.2)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(selected_features)
        plt.title("聚类中心雷达图")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    begin = pd.to_datetime("2023-02-04")
    end = pd.to_datetime("2024-02-04")
    inst_id = 'DOGE-USDT-SWAP'
    bar = '5m'

    drop_cols = [
        'inst_id_x', 'inst_id_y', 'ts', 'open', 'high', 'low', 'close',
        'vol', 'taker_sell', 'taker_buy', 'open_interest',
        'elite_long_short_ratio', 'elite_position_long_short_ratio',
        'all_long_short_ratio',
        'bullCondition', 'bearCondition', 'bullCondition_shift1',
        'bearCondition_shift1', 'buySignalRaw', 'sellSignalRaw',
        'buySignal', 'sellSignal', 'longFilter', 'shortFilter'
    ]

    ct = ClusterTrain(bar, begin, end, inst_id)
    ct.run(
        drop_columns=drop_cols,
        n_clusters=2,
        save=True
    )
