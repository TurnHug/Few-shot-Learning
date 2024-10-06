import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Dict, Any, List


class FeatureExtractor(nn.Module):
    """特征提取器，支持多种主干网络"""

    SUPPORTED_MODELS = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
        "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT, 512),
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
        "vgg16": (models.vgg16, models.VGG16_Weights.DEFAULT, 4096),
        "vgg19": (models.vgg19, models.VGG19_Weights.DEFAULT, 4096),
        "densenet121": (models.densenet121, models.DenseNet121_Weights.DEFAULT, 1024),
        "efficientnet_b0": (
            models.efficientnet_b0,
            models.EfficientNet_B0_Weights.DEFAULT,
            1280,
        ),
        "mobilenet_v2": (
            models.mobilenet_v2,
            models.MobileNet_V2_Weights.DEFAULT,
            1280,
        ),
    }

    def __init__(self, backbone_model: str = "resnet18", pretrained: bool = True):
        super().__init__()

        if backbone_model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {backbone_model}")

        model_fn, weights, self.feature_dim = self.SUPPORTED_MODELS[backbone_model]
        self.pretrained_model = model_fn(weights=weights if pretrained else None)

        # 构建backbone
        if "vgg" in backbone_model:
            self.backbone = self.pretrained_model.features
        else:
            self.backbone = nn.Sequential(*list(self.pretrained_model.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        if "vgg" in str(self.pretrained_model.__class__.__name__).lower():
            x = nn.AdaptiveAvgPool2d((7, 7))(x)
        return x.view(x.size(0), -1)


class BasePrototypeCalculator(nn.Module):
    """原型计算的基类"""

    def forward(self, support_features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SimplePrototypeCalculator(BasePrototypeCalculator):
    """简单平均原型计算器"""

    def forward(self, support_features: torch.Tensor) -> torch.Tensor:
        return support_features.mean(dim=1)


class WeightedPrototypeCalculator(BasePrototypeCalculator):
    """加权原型计算器"""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(feature_dim))

    def forward(self, support_features: torch.Tensor) -> torch.Tensor:
        weighted_features = support_features * self.weights
        return weighted_features.mean(dim=1)


class AttentionPrototypeCalculator(BasePrototypeCalculator):
    """注意力原型计算器"""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.ReLU(), nn.Linear(128, 1), nn.Softmax(dim=1)
        )

    def forward(self, support_features: torch.Tensor) -> torch.Tensor:
        attention_scores = self.attention(support_features)
        weighted_features = support_features * attention_scores
        return weighted_features.sum(dim=1)


class RelationModule(nn.Module):
    """关系模块，用于计算查询样本和原型之间的关系"""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: List[int] = [512, 256, 64],
        kernel_size: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()

        layers = []
        in_channels = input_channels

        # 卷积层
        for out_channels in hidden_channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Dropout2d(dropout),
                ]
            )
            in_channels = out_channels

        # 全连接层
        self.conv_net = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels[-1], 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_net(x)
        x = x.mean(dim=[2, 3])  # Global average pooling
        return self.fc(x)


class BaseFewShotModel(nn.Module):
    """小样本学习模型的基类"""

    def __init__(
        self,
        backbone_model: str = "resnet18",
        pretrained: bool = True,
        prototype_method: str = "simple",
    ):
        super().__init__()
        self.feature_extractor = FeatureExtractor(backbone_model, pretrained)

        # 初始化原型计算器
        feature_dim = self.feature_extractor.feature_dim
        if prototype_method == "weighted":
            self.prototype_calculator = WeightedPrototypeCalculator(feature_dim)
        elif prototype_method == "attention":
            self.prototype_calculator = AttentionPrototypeCalculator(feature_dim)
        else:
            self.prototype_calculator = SimplePrototypeCalculator()

    def compute_prototypes(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        n_way: int,
        n_shot: int,
    ) -> torch.Tensor:
        # 重新整理support_features为[n_way, n_shot, feature_dim]
        support_features = support_features.view(n_way, n_shot, -1)
        prototypes = []
        for i in range(n_way):
            # 直接获取第i类的所有support samples
            class_features = support_features[i : i + 1]  # [1, n_shot, feature_dim]
            prototype = self.prototype_calculator(class_features)
            prototypes.append(prototype)
        return torch.cat(prototypes, dim=0)  # [n_way, feature_dim]

    def predict(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        n_way: int,
    ) -> torch.Tensor:
        """
        进行预测并返回预测的类别

        Args:
            support_images: 支持集图像
            support_labels: 支持集标签
            query_images: 查询集图像
            n_way: 分类数量

        Returns:
            预测的类别标签
        """
        with torch.no_grad():
            # 进行预测
            logits = self.forward(support_images, support_labels, query_images, n_way)
            predictions = torch.argmax(logits, dim=1)  # 获取预测结果
        return predictions


class PrototypicalNetwork(BaseFewShotModel):
    """原型网络实现"""

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        n_way: int,
    ) -> torch.Tensor:
        # 计算n_shot
        n_shot = support_images.size(0) // n_way

        # 特征提取
        support_features = self.feature_extractor(
            support_images
        )  # [n_way*n_shot, feature_dim]
        query_features = self.feature_extractor(query_images)  # [n_query, feature_dim]

        # 计算原型
        prototypes = self.compute_prototypes(
            support_features, support_labels, n_way, n_shot
        )

        # 计算距离
        distances = torch.cdist(query_features, prototypes)
        return -distances


class MatchingNetwork(BaseFewShotModel):
    """匹配网络实现"""

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        n_way: int,
    ) -> torch.Tensor:
        n_shot = support_images.size(0) // n_way

        # 特征提取
        support_features = self.feature_extractor(support_images)
        query_features = self.feature_extractor(query_images)

        # 计算余弦相似度
        similarities = F.cosine_similarity(
            query_features.unsqueeze(1), support_features.unsqueeze(0), dim=2
        )

        # 转换标签为one-hot
        support_labels_one_hot = F.one_hot(support_labels, num_classes=n_way).float()

        # 计算logits
        logits = torch.matmul(similarities, support_labels_one_hot)
        return logits


class RelationNetwork(BaseFewShotModel):
    """关系网络实现"""

    def __init__(
        self,
        backbone_model: str = "resnet18",
        pretrained: bool = True,
        prototype_method: str = "simple",
        relation_module_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(backbone_model, pretrained, prototype_method)

        if relation_module_params is None:
            relation_module_params = {
                "input_channels": self.feature_extractor.feature_dim * 2,
                "hidden_channels": [512, 256, 64],
                "kernel_size": 3,
                "dropout": 0.5,
            }

        self.relation_module = RelationModule(**relation_module_params)

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        n_way: int,
    ) -> torch.Tensor:
        n_shot = support_images.size(0) // n_way

        # 特征提取
        support_features = self.feature_extractor(
            support_images
        )  # [n_way*n_shot, feature_dim]
        query_features = self.feature_extractor(query_images)  # [n_query, feature_dim]

        # 计算原型
        prototypes = self.compute_prototypes(
            support_features, support_labels, n_way, n_shot
        )

        # 准备关系对
        query_features = query_features.unsqueeze(1).repeat(
            1, n_way, 1
        )  # [n_query, n_way, feature_dim]
        prototypes = prototypes.unsqueeze(0).repeat(
            query_features.shape[0], 1, 1
        )  # [n_query, n_way, feature_dim]

        # 组合特征对
        relation_pairs = torch.cat(
            (query_features, prototypes), dim=2
        )  # [n_query, n_way, 2*feature_dim]
        relation_pairs = relation_pairs.view(
            -1, self.feature_extractor.feature_dim * 2, 1, 1
        )

        # 计算关系得分
        relations = self.relation_module(relation_pairs)
        return relations.view(-1, n_way)


def get_model(
    model_name: str,
    backbone_model: str = "resnet18",
    pretrained: bool = True,
    prototype_method: str = "simple",
    relation_module_params: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """
    获取指定的小样本学习模型

    Args:
        model_name: 模型名称 ('prototypical', 'matching', 或 'relation')
        backbone_model: 主干网络名称
        pretrained: 是否使用预训练权重
        prototype_method: 原型计算方法 ('simple', 'weighted', 或 'attention')
        relation_module_params: 关系模块的参数（仅用于关系网络）

    Returns:
        实例化的模型
    """
    if model_name == "prototypical":
        return PrototypicalNetwork(backbone_model, pretrained, prototype_method)
    elif model_name == "matching":
        return MatchingNetwork(backbone_model, pretrained, prototype_method)
    elif model_name == "relation":
        return RelationNetwork(
            backbone_model, pretrained, prototype_method, relation_module_params
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


# 损失函数
class PrototypicalLoss(nn.Module):
    """原型网络的损失函数"""

    def forward(self, logits: torch.Tensor, query_labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, query_labels)


class RelationLoss(nn.Module):
    """关系网络的损失函数"""

    def forward(
        self, relations: torch.Tensor, query_labels: torch.Tensor, n_way: int
    ) -> torch.Tensor:
        target = F.one_hot(query_labels, n_way).float()
        return F.mse_loss(relations, target)


if __name__ == "__main__":
    # 示例用法
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型配置
    model_config = {
        "backbone_model": "resnet18",
        "pretrained": True,
        "prototype_method": "attention",
        "relation_module_params": {
            "input_channels": 1024,  # 2 * feature_dim for relation pairs
            "hidden_channels": [512, 256, 64],
            "kernel_size": 3,
            "dropout": 0.5,
        },
    }

    # 创建模型
    model = get_model("relation", **model_config).to(device)

    # 模拟数据
    n_way, n_shot, n_query = 10, 5, 2  # 使用你的参数设置
    support_images = torch.randn(n_way * n_shot, 3, 224, 224).to(device)
    support_labels = torch.repeat_interleave(torch.arange(n_way), n_shot).to(device)
    query_images = torch.randn(n_way * n_query, 3, 224, 224).to(device)
    query_labels = torch.repeat_interleave(torch.arange(n_way), n_query).to(device)

    # 前向传播
    logits = model(support_images, support_labels, query_images, n_way)

    # 计算损失
    criterion = RelationLoss()
    loss = criterion(logits, query_labels, n_way)

    # 计算准确率
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == query_labels).float().mean()

    print(f"Loss: {loss.item():.4f}")
    print(f"Accuracy: {accuracy.item():.4f}")
