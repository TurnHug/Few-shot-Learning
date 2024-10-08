import argparse
import json
import logging
import os
from datetime import datetime

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloaders
from model import EnsembleLoss, RelationLoss, get_model, PrototypicalLoss

# 在文件开头添加以下代码来配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FewShotTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        ensemble=False,
        device="cuda",
        log_dir="runs",
    ):
        self.model = model.to(device)  # 将模型移动到指定设备
        self.train_loader = train_loader  # 训练数据加载器
        self.val_loader = val_loader  # 验证数据加载器
        self.device = device  # 设备类型
        self.ensemble = ensemble
        self.criterion = criterion
        self.optimizer = optimizer  # 优化器
        self.writer = SummaryWriter(log_dir)  # TensorBoard记录器
        self.scheduler = scheduler

    def train_epoch(self, epoch, n_way=10, k_shot=5, q_query=2, n_episodes=30):
        self.model.train()  # 设置模型为训练模式
        total_loss = 0  # 初始化总损失
        total_acc = 0  # 初始化总准确率
        if self.ensemble:
            # 记录名个模型的准确率
            model_accuracies = {name: 0.0 for name in self.model.models.keys()}

        with tqdm(range(n_episodes), desc=f"Epoch {epoch + 1} - train") as pbar:
            for episode in pbar:
                # 从训练集中采样一个episode
                episode_data = self.train_loader.dataset.dataset.sample_episode(
                    n_way=n_way, k_shot=k_shot, q_query=q_query
                )

                # 将数据移动到设备
                support_images = episode_data["support_images"].to(self.device)
                support_labels = episode_data["support_labels"].to(self.device)
                query_images = episode_data["query_images"].to(self.device)
                query_labels = episode_data["query_labels"].to(self.device)

                if self.ensemble:
                    # 获取各个模型的预测结果
                    individual_outputs = {}
                    individual_accuracies = {}
                    for name, model in self.model.models.items():
                        model.eval()  # 临时设置为评估模式
                        with torch.no_grad():
                            output = model(
                                support_images, support_labels, query_images, n_way
                            )
                            predictions = torch.argmax(output, dim=1)
                            acc = (predictions == query_labels).float().mean().item()
                            individual_accuracies[name] = acc
                            individual_outputs[name] = output
                        model.train()  # 恢复训练模式
                    # 更新模型权重
                    self.model.update_weights(individual_accuracies)
                    # 获取集成模型的预测结果
                    logits = self.model(
                        support_images, support_labels, query_images, n_way
                    )
                    # 计算损失
                    loss = self.criterion(logits, query_labels, individual_outputs)
                else:
                    # 前向传播计算logits
                    logits = self.model(
                        support_images, support_labels, query_images, n_way
                    )

                    try:
                        # 计算损失
                        loss = self.criterion(logits, query_labels)
                    except:
                        loss = self.criterion(logits, query_labels, n_way)

                self.optimizer.zero_grad()  # 清空梯度
                loss.backward()  # 反向传播
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )  # 添加梯度裁剪
                self.optimizer.step()  # 更新参数

                # 计算预测结果和准确率
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == query_labels).float().mean().item()

                total_loss += loss.item()  # 累加损失
                total_acc += accuracy  # 累加准确率
                if self.ensemble:

                    # 更新进度条信息
                    weights = self.model.get_weights()
                    weight_info = " ".join(
                        [f"{k[:3]}:{v:.2f}" for k, v in weights.items()]
                    )
                    pbar.set_postfix(
                        {
                            "train_loss": f"{loss.item():.4f}",
                            "train_acc": f"{accuracy:.4f}",
                            "weights": weight_info,
                        }
                    )

                else:

                    # 更新进度条
                    pbar.set_postfix(
                        {
                            "train_loss": f"{loss.item():.4f}",
                            "train_acc": f"{accuracy:.4f}",
                        }
                    )

        avg_loss = total_loss / n_episodes  # 计算平均损失
        avg_acc = total_acc / n_episodes  # 计算平均准确率

        # 记录训练损失和准确率
        self.writer.add_scalar("train/loss", avg_loss, epoch)
        self.writer.add_scalar("train/accuracy", avg_acc, epoch)
        if self.ensemble:
            # 记录每个模型的权重
            weights = self.model.get_weights()
            for name, weight in weights.items():
                self.writer.add_scalar(f"weights/{name}", weight, epoch)

        return avg_loss, avg_acc  # 返回平均损失和准确率

    def validate_epoch(self, epoch, n_val_episodes):
        self.model.eval()  # 设置模型为评估模式
        total_val_loss = 0
        total_val_acc = 0
        n_way = 10
        with torch.no_grad():
            with tqdm(
                range(n_val_episodes), desc=f"Epoch {epoch + 1} - validate"
            ) as pbar:
                for episode in pbar:
                    # 从验证集中采样一个episode
                    episode_data = self.val_loader.dataset.dataset.sample_episode(
                        n_way=10, k_shot=5, q_query=2
                    )

                    # 将数据移动到设备
                    support_images = episode_data["support_images"].to(self.device)
                    support_labels = episode_data["support_labels"].to(self.device)
                    query_images = episode_data["query_images"].to(self.device)
                    query_labels = episode_data["query_labels"].to(self.device)
                    if self.ensemble:
                        # 获取各个模型的预测结果
                        individual_outputs = {}
                        individual_accuracies = {}
                        for name, model in self.model.models.items():
                            model.eval()  # 临时设置为评估模式
                            with torch.no_grad():
                                output = model(
                                    support_images, support_labels, query_images, n_way
                                )
                                predictions = torch.argmax(output, dim=1)
                                acc = (
                                    (predictions == query_labels).float().mean().item()
                                )
                                individual_accuracies[name] = acc
                                individual_outputs[name] = output
                            model.train()  # 恢复训练模式
                        # 更新模型权重
                        self.model.update_weights(individual_accuracies)
                        # 获取集成模型的预测结果
                        logits = self.model(
                            support_images, support_labels, query_images, n_way
                        )
                        # 计算损失
                        # 计算损失
                        val_loss = self.criterion(
                            logits, query_labels, individual_outputs
                        )
                    else:
                        # 前向传播计算logits
                        logits = self.model(
                            support_images, support_labels, query_images, n_way
                        )
                        # 计算损失
                        try:
                            val_loss = self.criterion(logits, query_labels)
                        except:
                            val_loss = self.criterion(logits, query_labels, n_way)

                    # 计算预测结果和验证准确率
                    predictions = torch.argmax(logits, dim=1)
                    val_accuracy = (predictions == query_labels).float().mean().item()

                    total_val_loss += val_loss.item()  # 累加验证损失
                    total_val_acc += val_accuracy  # 累加验证准确率

                    # 更新进度条
                    pbar.set_postfix(
                        {
                            "val_loss": f"{val_loss.item():.4f}",
                            "val_acc": f"{val_accuracy:.4f}",
                        }
                    )

        avg_val_loss = total_val_loss / n_val_episodes  # 计算平均验证损失
        avg_val_acc = total_val_acc / n_val_episodes  # 计算平均验证准确率

        # 记录验证损失和准确率
        self.writer.add_scalar("val/loss", avg_val_loss, epoch)
        self.writer.add_scalar("val/accuracy", avg_val_acc, epoch)

        return avg_val_loss, avg_val_acc  # 返回平均验证损失和准确率

    def train(self, num_epochs, save_dir, n_episodes, n_val_episodes):
        os.makedirs(save_dir, exist_ok=True)  # 创建保存目录
        best_acc = 0  # 初始化最佳准确率
        patience = 0  # 初始化耐心计数

        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(epoch, n_episodes=n_episodes)
            val_loss, val_acc = self.validate_epoch(
                epoch, n_val_episodes=n_val_episodes
            )
            if self.scheduler is not None:
                self.scheduler.step()  # 更新学习率

            # 每5个epoch保存一次模型
            if (epoch + 1) % 5 == 0:
                if val_acc > best_acc:  # 如果当前验证准确率更好
                    best_acc = val_acc
                    torch.save(
                        self.model,
                        os.path.join(save_dir, "best_model.pth"),  # 保存最佳模型
                    )
                    patience = 0  # 重置耐心计数
                else:
                    patience += 1
                    if patience > 10:  # 如果超过耐心阈值
                        logger.info("Early stopping")
                        break

                logger.info(
                    f"Epoch [{epoch + 1}/{num_epochs}]: train loss: {train_loss:.4f}, train accuracy: {train_acc:.4f}, val loss: {val_loss:.4f}, val accuracy: {val_acc:.4f}"
                )

            # 简单保存模型
            torch.save(self.model, os.path.join(save_dir, "latest_model.pth"))

        self.writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="小样本学习训练")
    parser.add_argument("--train_dir", type=str, required=True, help="训练数据集路径")
    parser.add_argument(
        "--model",
        type=str,
        default="ensemble",
        choices=["prototypical", "matching", "relation", "ensemble"],
        help="选择模型",  # 模型选择
    )

    parser.add_argument(
        "--backbone_model",
        type=str,
        default="resnet18",
        choices=[
            "resnet18",
            "resnet34",
            "densenet121",
            "efficientnet_b0",
            "mobilenet_v2",
        ],
        help="选择特征提取预训练模型",
    )

    parser.add_argument(
        "--prototype_method",
        type=str,
        default="weighted",
        choices=["simple", "weighted", "attention"],
        help="选择分类模型的原型计算方法",
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=30,
        choices=[60, 80, 100],
        help="每个epoch训练的episode数量",
    )
    parser.add_argument(
        "--n_val_episodes", type=int, default=5, help="每个epoch验证的episode数量"
    )
    parser.add_argument("--num_epochs", type=int, default=30, help="训练轮数")
    parser.add_argument(
        "--optimizer", type=str, default="sgd", choices=["adam", "sgd"], help="优化器"
    )
    parser.add_argument("--learning_rate", type=float, default=0.001, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument(
        "--lr_scheduler",
        type=bool,
        default=False,
        choices=[True, False],
        help="学习率调整",
    )

    parser.add_argument("--n_way", type=int, default=10, help="分类的类别数")
    parser.add_argument(
        "--k_shot", type=int, default=5, help="支持集中每个类别的样本数"
    )
    parser.add_argument(
        "--q_query", type=int, default=2, help="查询集中每个类别的样本数"
    )
    parser.add_argument(
        "--val_size", type=float, default=0.35, help="划分训练集和验证集的比例"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="使用的设备 (cuda/cpu)"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(json.dumps(vars(args), indent=4))  # 记录参数信息

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")  # 获取当前时间戳
    exp_dir = os.path.join("experiments", timestamp)  # 创建实验目录
    os.makedirs(exp_dir, exist_ok=True)  # 创建目录

    train_loader, val_loader = get_dataloaders(
        args.train_dir, "", args.val_size, is_train=True
    )
    # 创建模型配置
    model_config = {
        "backbone_model": args.backbone_model,
        "pretrained": True,
        "prototype_method": args.prototype_method,
        "relation_module_params": {
            "input_channels": 1024,  # 2 * feature_dim for relation pairs
            "hidden_channels": [512, 256, 64],
            "kernel_size": 3,
            "dropout": 0.3,
        },
        "temperature": 0.5,  # 可以通过参数控制
        "momentum": 0.9,  # 可以通过参数控制
    }
    if args.model == "prototypical" or args.model == "matching":
        criterion = PrototypicalLoss()
    elif args.model == "relation":
        criterion = RelationLoss()
    elif args.model == "ensemble":
        criterion = EnsembleLoss()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = get_model(args.model, **model_config)

    if args.optimizer == "adam":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None

    if args.model == "ensemble":
        ensemble = True
    else:
        ensemble = False

    trainer = FewShotTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        ensemble=ensemble,
        device=args.device,
        log_dir=os.path.join(exp_dir, "logs"),  # 日志目录
    )

    trainer.train(
        num_epochs=args.num_epochs,
        save_dir=os.path.join(exp_dir, "checkpoints"),  # 保存检查点目录
        n_episodes=args.n_episodes,
        n_val_episodes=args.n_val_episodes,
    )
    args.criterion = str(trainer.criterion)  # 获取损失函数
    if args.lr_scheduler:

        args.lr_scheduler = str(
            trainer.scheduler.state_dict()
        )  # 获取学习率调度器的初始化参数
        args.is_scheduler = True
    else:
        args.lr_scheduler = None
        args.is_scheduler = False
    if args.model != "ensemble":
        args.feature_dim = str(
            trainer.model.feature_extractor.feature_dim
        )  # 获取特征维度

    with open(os.path.join(exp_dir, "model_info.json"), "w") as f:
        json.dump(vars(args), f, indent=4)  # 保存模型信息
    logger.info(f"experiment finished, saved to: {exp_dir}")


if __name__ == "__main__":
    main()
