import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights


class FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # 加载预训练的ResNet18模型
        self.pretrained_model = models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        # 去掉最后一层，保留特征提取部分
        self.backbone = nn.Sequential(*list(self.pretrained_model.children())[:-1])
        self.feature_dim = 512  # 特征维度

    def forward(self, x):
        # 前向传播，提取特征
        x = self.backbone(x)
        return x.view(x.size(0), -1)  # 将特征展平


class PrototypicalNetwork(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # 初始化特征提取器
        self.feature_extractor = FeatureExtractor(pretrained=pretrained)

    def forward(self, support_images, support_labels, query_images, n_way):
        """
            原型网络的前向传递。

                参数
                ----------
                support_images : torch.Tensor
                    支持集的输入图像
                support_labels : torch.Tensor
                    支持集的标签
                query_images : torch.Tensor
                    查询集的输入图像
                n_way : int
                    事件中的类数

                返回
                -------
                张量
                    查询图像到原型的负距离
        """
        # 提取支持集和查询集的特征
        support_features = self.feature_extractor(support_images)
        query_features = self.feature_extractor(query_images)
        # 计算原型
        prototypes = self.compute_prototypes(support_features, support_labels, n_way)
        # 计算距离
        distances = self.compute_distances(query_features, prototypes)
        return -distances  # 返回负距离

    def compute_prototypes(self, support_features, support_labels, n_way):
        prototypes = []
        for i in range(n_way):
            # 创建掩码，选择对应类别的特征
            mask = support_labels == i
            # 计算原型
            prototype = support_features[mask].mean(dim=0)
            prototypes.append(prototype)
        return torch.stack(prototypes)  # 返回原型张量

    def compute_distances(self, query_features, prototypes):
        # 计算查询特征与原型之间的距离
        return torch.cdist(query_features, prototypes)

    def predict(self, support_images, support_labels, query_images, n_way):
        with torch.no_grad():
            # 进行预测
            logits = self.forward(support_images, support_labels, query_images, n_way)
            predictions = torch.argmax(logits, dim=1)  # 获取预测结果
        return predictions


class MatchingNetwork(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # 初始化特征提取器
        self.feature_extractor = FeatureExtractor(pretrained=pretrained)

    def forward(self, support_images, support_labels, query_images, n_way):
        # 提取支持集和查询集的特征
        support_features = self.feature_extractor(support_images)
        query_features = self.feature_extractor(query_images)
        
        # 计算相似度
        similarities = F.cosine_similarity(query_features.unsqueeze(1), support_features.unsqueeze(0), dim=2)
        
        # 将支持集标签转换为one-hot编码
        support_labels_one_hot = F.one_hot(support_labels, num_classes=n_way).float()
        # 计算logits
        logits = torch.matmul(similarities, support_labels_one_hot)
        
        return logits

    def predict(self, support_images, support_labels, query_images, n_way):
        with torch.no_grad():
            # 进行预测
            logits = self.forward(support_images, support_labels, query_images, n_way)
            predictions = torch.argmax(logits, dim=1)  # 获取预测结果
        return predictions


class RelationNetwork(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.feature_extractor = FeatureExtractor(pretrained=pretrained)
        self.relation_module = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, support_images, support_labels, query_images, n_way):
        support_features = self.feature_extractor(support_images)
        query_features = self.feature_extractor(query_images)
        
        support_features = support_features.view(n_way, -1, support_features.shape[-1])
        support_prototypes = support_features.mean(dim=1)
        
        query_features = query_features.unsqueeze(1).repeat(1, n_way, 1)
        support_prototypes = support_prototypes.unsqueeze(0).repeat(query_features.shape[0], 1, 1)
        
        relation_pairs = torch.cat((query_features, support_prototypes), dim=2)
        relation_pairs = relation_pairs.view(-1, 1024, 1, 1)
        
        relations = self.relation_module(relation_pairs).view(-1, n_way)
        return relations

    def predict(self, support_images, support_labels, query_images, n_way):
        with torch.no_grad():
            logits = self.forward(support_images, support_labels, query_images, n_way)
            predictions = torch.argmax(logits, dim=1)
        return predictions


class PrototypicalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, query_labels):
        # 计算交叉熵损失
        return F.cross_entropy(logits, query_labels)


def get_model(model_name, pretrained=True):
    # 根据模型名称返回相应的模型
    if model_name == 'prototypical':
        return PrototypicalNetwork(pretrained=pretrained)
    elif model_name == 'matching':
        return MatchingNetwork(pretrained=pretrained)
    elif model_name == 'relation':
        return RelationNetwork(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


if __name__ == "__main__":
    model = PrototypicalNetwork()
    support_images = torch.randn(5 * 2, 3, 224, 224)  # 随机生成支持集图像
    support_labels = torch.rand(10)  # 随机生成支持集标签
    query_images = torch.randn(5 * 3, 3, 224, 224)  # 随机生成查询集图像
    result = model.predict(support_images, support_labels, query_images, 5)  # 进行预测
    print(result)  # 打印预测结果
