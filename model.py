import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Dict, Any, List
from collections import OrderedDict

class FeatureExtractor(nn.Module):
    """特征提取器，支持多种主干网络"""

    SUPPORTED_MODELS = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
        "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT, 512),
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
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
        self.backbone_name = backbone_model
        self.pretrained_model = model_fn(weights=weights if pretrained else None)

        # 构建backbone
        if "vgg" in backbone_model:
            self.backbone = self.pretrained_model.features
        elif "densenet" in backbone_model:
            # DenseNet特殊处理
            self.backbone = nn.Sequential(
                self.pretrained_model.features,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        elif "mobilenet" in backbone_model:
            # MobileNetV2特殊处理
            features = self.pretrained_model.features
            self.backbone = nn.Sequential(
                features,
                nn.AdaptiveAvgPool2d((1, 1))
            )
        else:
            self.backbone = nn.Sequential(*list(self.pretrained_model.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
    
        
        # 处理特征形状
        if len(x.shape) == 4:
            # 如果已经是4维张量，确保最后两维是1x1
            if x.shape[2] != 1 or x.shape[3] != 1:
                x = nn.AdaptiveAvgPool2d((1, 1))(x)
        else:
            # 如果不是4维张量，转换为[batch_size, channels, 1, 1]格式
            x = x.view(x.size(0), -1, 1, 1)
        
        return x


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
        n_shot = support_images.size(0) // n_way

        # 特征提取
        support_features = self.feature_extractor(support_images)  # [n_way*n_shot, feature_dim, 1, 1]
        query_features = self.feature_extractor(query_images)      # [n_query, feature_dim, 1, 1]

        # 确保特征维度正确
        if support_features.dim() == 4:
            support_features = support_features.squeeze(-1).squeeze(-1)  # [n_way*n_shot, feature_dim]
        if query_features.dim() == 4:
            query_features = query_features.squeeze(-1).squeeze(-1)      # [n_query, feature_dim]

        # 计算原型
        prototypes = self.compute_prototypes(
            support_features, support_labels, n_way, n_shot
        )  # [n_way, feature_dim]

        # 计算距离
        distances = torch.cdist(query_features, prototypes)  # [n_query, n_way]
        return -distances  # 返回负距离作为相似度得分


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
        n_query = query_images.size(0)

        # 特征提取
        support_features = self.feature_extractor(support_images)  # [n_way*n_shot, feature_dim, 1, 1]
        query_features = self.feature_extractor(query_images)      # [n_query, feature_dim, 1, 1]

        # 确保特征维度正确
        if support_features.dim() == 4:
            support_features = support_features.squeeze(-1).squeeze(-1)  # [n_way*n_shot, feature_dim]
        if query_features.dim() == 4:
            query_features = query_features.squeeze(-1).squeeze(-1)      # [n_query, feature_dim]

        # 计算余弦相似度
        support_features_norm = F.normalize(support_features, p=2, dim=1)  # L2 归一化
        query_features_norm = F.normalize(query_features, p=2, dim=1)     # L2 归一化
        
        # 计算相似度矩阵 [n_query, n_way*n_shot]
        similarities = torch.mm(query_features_norm, support_features_norm.t())

        # 转换标签为 one-hot 编码 [n_way*n_shot, n_way]
        support_labels_one_hot = F.one_hot(support_labels, num_classes=n_way).float()

        # 计算加权投票
        # [n_query, n_way*n_shot] × [n_way*n_shot, n_way] = [n_query, n_way]
        logits = torch.matmul(similarities, support_labels_one_hot)

        # 对每个查询样本的相似度进行归一化
        logits = logits / n_shot  # 可选：按照 shot 数量归一化

        return logits

    def compute_attention(self, query_features: torch.Tensor, support_features: torch.Tensor) -> torch.Tensor:
        """计算注意力权重（可选的附加功能）"""
        energy = torch.bmm(
            query_features.unsqueeze(1),
            support_features.transpose(1, 2)
        ).squeeze(1)
        return F.softmax(energy, dim=1)


# 2. RelationModule优化
class RelationModule(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: List[int] = [512, 256, 128, 64],
        kernel_size: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        # 添加注意力机制
        self.channel_attention = nn.Sequential(
            nn.Linear(input_channels, input_channels // 16),
            nn.ReLU(),
            nn.Linear(input_channels // 16, input_channels),
            nn.Sigmoid(),
        )

        layers = []
        in_channels = input_channels

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

        self.residual = nn.Conv2d(input_channels, hidden_channels[-1], 1)
        self.conv_net = nn.Sequential(*layers)

        # 修改最终分类头
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels[-1], hidden_channels[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels[-1], hidden_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels[-1] // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 应用通道注意力
        attention = self.channel_attention(x.squeeze(-1).squeeze(-1))
        x = x * attention.unsqueeze(-1).unsqueeze(-1)

        residual = self.residual(x)
        x = self.conv_net(x)
        x = x + residual
        x = F.adaptive_avg_pool2d(x, 1)
        return self.fc(x.view(x.size(0), -1))


class RelationNetwork(BaseFewShotModel):
    def __init__(
        self,
        backbone_model: str = "resnet18",
        pretrained: bool = True,
        prototype_method: str = "simple",
        relation_module_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(backbone_model, pretrained, prototype_method)

        # 获取特征维度
        self.feature_dim = self.feature_extractor.feature_dim
        
        # 修改特征处理器，确保能处理所有维度的输入
        self.feature_processor = nn.Sequential(
            nn.Conv2d(self.feature_dim, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        if relation_module_params is None:
            relation_module_params = {
                "input_channels": 1024,  # 2 * 512 for relation pairs
                "hidden_channels": [512, 256, 128, 64],
                "kernel_size": 3,
                "dropout": 0.3,
            }

        self.relation_module = RelationModule(**relation_module_params)

        # 添加注意力机制
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

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

        # 确保特征维度正确
        if support_features.dim() < 4:
            support_features = support_features.unsqueeze(-1).unsqueeze(-1)
        if query_features.dim() < 4:
            query_features = query_features.unsqueeze(-1).unsqueeze(-1)

        # 特征处理，统一转换为512维
        support_features = self.feature_processor(support_features)  # [n_way*n_shot, 512, 1, 1]
        query_features = self.feature_processor(query_features)      # [n_query, 512, 1, 1]

        # 添加注意力权重
        support_weights = self.attention(support_features.squeeze(-1).squeeze(-1))
        support_features = support_features * support_weights.unsqueeze(-1).unsqueeze(-1)

        # 计算原型
        support_features_flat = support_features.squeeze(-1).squeeze(-1)
        prototypes = self.compute_prototypes(support_features_flat, support_labels, n_way, n_shot)
        prototypes = prototypes.view(n_way, 512, 1, 1)

        # 准备关系对
        n_query = query_features.size(0)
        query_features = query_features.unsqueeze(1).repeat(1, n_way, 1, 1, 1)
        prototypes = prototypes.unsqueeze(0).repeat(n_query, 1, 1, 1, 1)

        # 组合特征对
        relation_pairs = torch.cat((query_features, prototypes), dim=2).view(-1, 1024, 1, 1)

        # 计算关系得分
        relations = self.relation_module(relation_pairs)
        return relations.view(-1, n_way)


class DynamicWeightLayer(nn.Module):
    """动态权重层，用于自适应调整各个模型的权重"""
    def __init__(self, n_models: int, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        # 初始化为相等权重
        self.weights = nn.Parameter(torch.ones(n_models) / n_models)
        
    def forward(self) -> torch.Tensor:
        # 使用softmax确保权重和为1
        return F.softmax(self.weights / self.temperature, dim=0)

class MajorityVotingEnsembleNet(nn.Module):
    """少数服从多数集成模型，结合原型网络、匹配网络和关系网络"""
    def __init__(
        self,
        backbone_model: str = "resnet18",
        pretrained: bool = True,
        prototype_method: str = "weighted",
        relation_module_params: Optional[Dict[str, Any]] = None,
        temperature: float = 1.0,
        momentum: float = 0.9  # 用于平滑权重更新
    ):
        super().__init__()
        
        # 初始化三个不同的基础模型
        self.models = nn.ModuleDict({
            'prototypical': PrototypicalNetwork(backbone_model, pretrained, prototype_method),
            'matching': MatchingNetwork(backbone_model, pretrained, prototype_method),
            'relation': RelationNetwork(backbone_model, pretrained, prototype_method, relation_module_params)
        })
        
        # 初始化动态权重层
        self.weight_layer = DynamicWeightLayer(len(self.models), temperature)
        
        # 记录每个模型的移动平均准确率
        self.model_accuracies = {name: 0.0 for name in self.models.keys()}
        self.momentum = momentum
        
    def update_weights(self, accuracies: Dict[str, float]):
        """根据各个模型的准确率更新权重"""
        for name, acc in accuracies.items():
            self.model_accuracies[name] = (
                self.momentum * self.model_accuracies[name] +
                (1 - self.momentum) * acc
            )
        
        accuracy_tensor = torch.tensor(
            [self.model_accuracies[name] for name in self.models.keys()]
        )

        # 使用softmax进行归一化
        accuracy_tensor = F.softmax(accuracy_tensor, dim=0)
        
        with torch.no_grad():
            self.weight_layer.weights.copy_(accuracy_tensor)
    
    def get_weights(self) -> Dict[str, float]:
        """获取当前各个模型的权重"""
        weights = self.weight_layer().detach()
        return {name: weight.item() for name, weight in zip(self.models.keys(), weights)}
    
    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        n_way: int
    ) -> torch.Tensor:
        # 获取每个模型的预测结果
        model_outputs = {name:model(support_images, support_labels, query_images, n_way) for name, model in self.models.items()}
        
        # 获取每个模型的独立预测
        model_predictions = {name: torch.argmax(output, dim=1) for name, output in model_outputs.items()}
        
        # 获取模型的权重
        weights = self.weight_layer()
        
        # 初始化 vote_counts，维度为 (batch_size, n_way)
        vote_counts = torch.zeros((query_images.size(0), n_way), device=query_images.device)
        
        # 进行加权投票
        for i, (name, predictions) in enumerate(model_predictions.items()):
            for batch_idx, predicted_class in enumerate(predictions):
                vote_counts[batch_idx, predicted_class] += weights[i]
        
        # 返回累积的 vote_counts 作为最终结果
        return vote_counts

    
    def predict(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        n_way: int
    ) -> torch.Tensor:
        """使用少数服从多数机制进行预测"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(support_images, support_labels, query_images, n_way)
            predictions = torch.argmax(logits, dim=1)  # 获取预测结果
        return predictions

class EnsembleLoss(nn.Module):
    """集成模型的损失函数"""
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(
        self,
        ensemble_output: torch.Tensor,
        target: torch.Tensor,
        individual_outputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        # 计算主要的交叉熵损失
        main_loss = self.ce_loss(ensemble_output, target)
        
        # 如果提供了个别模型的输出，添加一致性损失
        if individual_outputs is not None:
            consistency_loss = 0
            n_models = len(individual_outputs)
            
            # 计算所有模型输出之间的KL散度
            for i, (name1, output1) in enumerate(individual_outputs.items()):
                for name2, output2 in list(individual_outputs.items())[i+1:]:
                    kl_div = F.kl_div(
                        F.log_softmax(output1 / self.temperature, dim=1),
                        F.softmax(output2 / self.temperature, dim=1),
                        reduction='batchmean'
                    )
                    consistency_loss += kl_div
            
            # 将一致性损失添加到主损失中
            total_loss = main_loss + 0.05 * consistency_loss
            return total_loss
        
        return main_loss

def get_ensemble_model(
    backbone_model: str = "resnet18",
    pretrained: bool = True,
    prototype_method: str = "weighted",
    relation_module_params: Optional[Dict[str, Any]] = None,
    temperature: float = 1.0,
    momentum: float = 0.9
) -> nn.Module:
    """创建集成模型的工厂函数"""
    return MajorityVotingEnsembleNet(
        backbone_model,
        pretrained,
        prototype_method,
        relation_module_params,
        temperature,
        momentum
    )

def get_model(
    model_name: str,
    backbone_model: str = "resnet18",
    pretrained: bool = True,
    prototype_method: str = "simple",
    relation_module_params: Optional[Dict[str, Any]] = None,
    temperature: float = 1.0,
    momentum: float = 0.9
) -> nn.Module:
    """
    获取指定的小样本学习模型

    Args:
        model_name: 模型名称 ('prototypical', 'matching', 'relation', 或 'ensemble')
        backbone_model: 主干网络名称
        pretrained: 是否使用预训练权重
        prototype_method: 原型计算方法 ('simple', 'weighted', 或 'attention')
        relation_module_params: 关系模块的参数（仅用于关系网络）
        temperature: 温度参数（仅用于集成模型）
        momentum: 动量参数（仅用于集成模型）

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
    elif model_name == "ensemble":
        return get_ensemble_model(
            backbone_model, pretrained, prototype_method, relation_module_params, temperature, momentum
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


# 损失函数
class PrototypicalLoss(nn.Module):
    """原型网络的损失函数"""

    def forward(self, logits: torch.Tensor, query_labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, query_labels)


class RelationLoss(nn.Module):
    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self, relations: torch.Tensor, query_labels: torch.Tensor, n_way: int
    ) -> torch.Tensor:
        batch_size = relations.size(0)

        # 创建one-hot标签
        target = F.one_hot(query_labels, n_way).float()

        # MSE Loss
        mse_loss = F.mse_loss(relations, target)

        # 计算对比损失
        pos_mask = target == 1
        neg_mask = target == 0

        contrast_loss = 0.0

        # 对每个查询样本分别计算对比损失
        for i in range(batch_size):
            sample_relations = relations[i]
            sample_pos = sample_relations[pos_mask[i]]
            sample_neg = sample_relations[neg_mask[i]]

            if sample_pos.numel() > 0 and sample_neg.numel() > 0:
                # 将正样本扩展到与负样本相同的大小
                sample_pos = sample_pos.expand(sample_neg.size(0))

                # 计算当前样本的对比损失
                sample_contrast_loss = torch.mean(
                    torch.relu(sample_neg - sample_pos + self.margin)
                )
                contrast_loss += sample_contrast_loss

        # 平均对比损失
        if batch_size > 0:
            contrast_loss = contrast_loss / batch_size

        # 计算交叉熵损失
        logits = torch.log(relations / (1 - relations + 1e-7))
        ce_loss = self.ce_loss(logits, query_labels)

        # 组合损失
        total_loss = mse_loss + 0.1 * contrast_loss + ce_loss
        return total_loss


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
        "temperature": 1.0,
        "momentum": 0.9
    }

    # 创建模型
    model = get_model("ensemble", **model_config).to(device)

    # 模拟数据
    n_way, n_shot, n_query = 10, 5, 2  # 使用你的参数设置
    support_images = torch.randn(n_way * n_shot, 3, 224, 224).to(device)
    support_labels = torch.repeat_interleave(torch.arange(n_way), n_shot).to(device)
    query_images = torch.randn(n_way * n_query, 3, 224, 224).to(device)
    query_labels = torch.repeat_interleave(torch.arange(n_way), n_query).to(device)
    print(query_labels)

    # 前向传播
    logits = model(support_images, support_labels, query_images, n_way)
    print(logits)

    # 计算损失
    criterion = EnsembleLoss()
    loss = criterion(logits, query_labels)

    # 计算准确率
    predictions = torch.argmax(logits, dim=1)
    print(predictions)
    accuracy = (predictions == query_labels).float().mean()

    print(f"Loss: {loss.item():.4f}")
    print(f"Accuracy: {accuracy.item():.4f}")
