import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights


class FeatureExtractor(nn.Module):
    def __init__(self, backbone_model='resnet18', pretrained=True):
        super().__init__()
        # 根据传入的 backbone 模型名称初始化模型
        if backbone_model == 'resnet18':
            self.pretrained_model = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = 512
        elif backbone_model == 'resnet34':
            self.pretrained_model = models.resnet34(
                weights=models.ResNet34_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = 512
        elif backbone_model == 'resnet50':
            self.pretrained_model = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = 2048
        elif backbone_model == 'vgg16':
            self.pretrained_model = models.vgg16(
                weights=models.VGG16_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = 4096
            # 去掉分类器部分，保留特征提取层
            self.backbone = self.pretrained_model.features
        elif backbone_model == 'vgg19':
            self.pretrained_model = models.vgg19(
                weights=models.VGG19_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = 4096
            self.backbone = self.pretrained_model.features
        elif backbone_model == 'densenet121':
            self.pretrained_model = models.densenet121(
                weights=models.DenseNet121_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = 1024
        elif backbone_model == 'efficientnet_b0':
            self.pretrained_model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = 1280
        elif backbone_model == 'mobilenet_v2':
            self.pretrained_model = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = 1280  # MobileNetV2输出特征维度
        else:
            raise ValueError(f"Unsupported model: {backbone_model}")

        # 对于 VGG，主干已经设置好了。对于其他模型删除最后一层
        if 'vgg' not in backbone_model:
            self.backbone = nn.Sequential(*list(self.pretrained_model.children())[:-1])

    def forward(self, x):
        # 前向传递以提取特征
        x = self.backbone(x)
        if 'vgg' in self.pretrained_model.__class__.__name__.lower():
            # 对于 VGG 模型，在返回之前展平
            x = nn.AdaptiveAvgPool2d((7, 7))(x)
            return x.view(x.size(0), -1)
        return x.view(x.size(0), -1)


class PrototypicalNetwork(nn.Module):
    def __init__(self, backbone_model='resnet18', pretrained=True):
        super().__init__()
        # 初始化特征提取器
        self.feature_extractor = FeatureExtractor(backbone_model=backbone_model, pretrained=pretrained)

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
    def __init__(self, backbone_model='resnet18', pretrained=True):
        super().__init__()
        # 初始化特征提取器
        self.feature_extractor = FeatureExtractor(backbone_model=backbone_model, pretrained=pretrained)

    def forward(self, support_images, support_labels, query_images, n_way):
        # 提取支持集和查询集的特征
        support_features = self.feature_extractor(support_images)
        query_features = self.feature_extractor(query_images)

        # 计算相似度
        similarities = F.cosine_similarity(
            query_features.unsqueeze(1), support_features.unsqueeze(0), dim=2
        )

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
    def __init__(self, backbone_model='resnet18', pretrained=True):
        super().__init__()
        self.feature_extractor = FeatureExtractor(backbone_model=backbone_model, pretrained=pretrained)
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
            nn.Linear(8, 1),
        )

    def forward(self, support_images, support_labels, query_images, n_way):
        support_features = self.feature_extractor(support_images)
        query_features = self.feature_extractor(query_images)

        support_features = support_features.view(n_way, -1, support_features.shape[-1])
        support_prototypes = support_features.mean(dim=1)

        query_features = query_features.unsqueeze(1).repeat(1, n_way, 1)
        support_prototypes = support_prototypes.unsqueeze(0).repeat(
            query_features.shape[0], 1, 1
        )

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


def get_model(model_name, backbone_model='resnet18', pretrained=True):
    # 根据模型名称返回相应的模型
    if model_name == "prototypical":
        return PrototypicalNetwork(backbone_model=backbone_model, pretrained=pretrained)
    elif model_name == "matching":
        return MatchingNetwork(backbone_model=backbone_model, pretrained=pretrained)
    elif model_name == "relation":
        return RelationNetwork(backbone_model= backbone_model, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


if __name__ == "__main__":
    model = PrototypicalNetwork()
    support_images = torch.randn(5 * 2, 3, 224, 224)  # 随机生成支持集图像
    support_labels = torch.rand(10)  # 随机生成支持集标签
    query_images = torch.randn(5 * 3, 3, 224, 224)  # 随机生成查询集图像
    result = model.predict(support_images, support_labels, query_images, 5)  # 进行预测
    print(result)  # 打印预测结果
