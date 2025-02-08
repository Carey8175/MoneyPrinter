import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from system_code.core.modules.print_system import PrintSystem, logger
from system_code.core.modules.position import HoldingGroup


class ClusterTrain(PrintSystem):
    def __init__(self, bar, train_begin_date, train_end_date, inst_id):
        super(ClusterTrain, self).__init__(
            bar=bar,
            begin=train_begin_date,
            end=train_end_date,
            inst_id=inst_id
        )

    def run(self):
        logger.info(f"[Training] {str(self)}")

        data = self.fetch_data()
        # holding_groups = self.open_module.open_position(data)

    def train_cluster(self, group: HoldingGroup, n_clusters: int = 2, save: bool = True, feature_columns: list = None):
        """
        Clustering the holding group
        Args:
            group: 将这个group进行聚类
            n_clusters: 聚类的数量
            save: 是否保存聚类结果
            feature_columns: 聚类时需要删除的列

        Returns:

        """
        if feature_columns is None:
            feature_columns = group[0].data.columns

        # 确保所有特征列都在 data 中；若没有 volume，可自行去掉或替换
        available_features = [col for col in feature_columns if col in group[0].data.columns]

        if len(available_features) != len(feature_columns):
            logger.warning(f"feature_columns 中有些列不在数据中，已经被删除！")

        # 构建一个列表，用于存放聚类所需的特征行
        X_list = []
        for hp, idx in group:
            # 提取该 signal bar 的特征值
            row = data.loc[idx, available_features]
            # row 可能是 Series，需要先变成 np.array 或者 list
            X_list.append(row.values)

        X = np.array(X_list, dtype=float)

        # 数据清洗（如缺失值填充）与缩放（可视需要选择）
        # 这里简单做个 fillna(0) + 标准化示例
        X = np.nan_to_num(X, nan=0.0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 4.2. 聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X_scaled)

        # 4.3. 按照聚类结果，将 HoldingPeriod 分组
        holding_groups = [HoldingGroup() for _ in range(n_clusters)]

        # 遍历聚类标签，把对应的 HoldingPeriod 加入对应的组
        for hp, label in zip(all_hp, labels):
            holding_groups[label].add(hp[0])

        # 统计、更新每个HoldingGroup信息（可选）
        for group in holding_groups:
            group.update()

        # 返回聚类后的多个 HoldingGroup
        return holding_groups