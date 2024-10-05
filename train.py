import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import json
import os
import argparse
from datetime import datetime
from dataset import get_dataloaders
from model import get_model, PrototypicalLoss
import torch.optim.lr_scheduler as lr_scheduler
import logging

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
        lr=0.01,
        device="cuda",
        log_dir="runs",
    ):
        self.model = model.to(device)  # 将模型移动到指定设备
        self.train_loader = train_loader  # 训练数据加载器
        self.val_loader = val_loader  # 验证数据加载器
        self.device = device  # 设备类型
        self.criterion = PrototypicalLoss()  # 损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)  # 优化器
        self.writer = SummaryWriter(log_dir)  # TensorBoard记录器
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=25, gamma=0.5
        )  # 学习率调度器

    def train_epoch(self, epoch, n_way=10, k_shot=5, q_query=2, n_episodes=30):
        self.model.train()  # 设置模型为训练模式
        total_loss = 0  # 初始化总损失
        total_acc = 0  # 初始化总准确率

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

                # 前向传播计算logits
                logits = self.model(support_images, support_labels, query_images, n_way)

                # 计算损失
                loss = self.criterion(logits, query_labels)

                self.optimizer.zero_grad()  # 清空梯度
                loss.backward()  # 反向传播
                self.optimizer.step()  # 更新参数

                # 计算预测结果和准确率
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == query_labels).float().mean().item()

                total_loss += loss.item()  # 累加损失
                total_acc += accuracy  # 累加准确率

                # 更新进度条
                pbar.set_postfix(
                    {"train_loss": f"{loss.item():.4f}", "train_acc": f"{accuracy:.4f}"}
                )

        avg_loss = total_loss / n_episodes  # 计算平均损失
        avg_acc = total_acc / n_episodes  # 计算平均准确率

        # 记录训练损失和准确率
        self.writer.add_scalar("train/loss", avg_loss, epoch)
        self.writer.add_scalar("train/accuracy", avg_acc, epoch)

        return avg_loss, avg_acc  # 返回平均损失和准确率


    def validate_epoch(self, epoch, n_val_episodes):
        self.model.eval()  # 设置模型为评估模式
        total_val_loss = 0
        total_val_acc = 0

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

                    # 前向传播计算logits
                    logits = self.model(
                        support_images, support_labels, query_images, n_way=10
                    )

                    # 计算验证损失
                    val_loss = self.criterion(logits, query_labels)

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

        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(epoch, n_episodes=n_episodes)
            val_loss, val_acc = self.validate_epoch(
                epoch, n_val_episodes=n_val_episodes
            )

            self.scheduler.step()  # 更新学习率

            # 每5个epoch保存一次模型
            if (epoch + 1) % 5 == 0:
                if val_acc > best_acc:  # 如果当前验证准确率更好
                    best_acc = val_acc
                    torch.save(
                        self.model,
                        os.path.join(save_dir, "best_model.pth"),  # 保存最佳模型
                    )

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
        default="prototypical",
        choices=["prototypical", "matching", "relation"],
        help="选择模型",  # 模型选择
    )
    
    parser.add_argument(
        "--backbone_model",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50", "vgg16", "vgg19","densenet121","efficientnet_b0", "mobilenet_v2"],
        help="选择特征提取预训练模型",
    )
    parser.add_argument("--learning_rate", type=float, default=0.001, help="学习率")
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮数")
    parser.add_argument(
        "--n_episodes", type=int, default=10, help="每个epoch训练的episode数量"
    )
    parser.add_argument(
        "--n_val_episodes", type=int, default=5, help="每个epoch验证的episode数量"
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

    model = get_model(args.model, args.backbone_model, pretrained=True)

    trainer = FewShotTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.learning_rate,
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
    args.optimizer = str(trainer.optimizer)  # 获取优化器
    args.lr_scheduler = str(trainer.scheduler.state_dict())  # 获取学习率调度器的初始化参数
    args.feature_dim = str(trainer.model.feature_extractor.feature_dim)  # 获取特征维度


    with open(os.path.join(exp_dir, "model_info.json"), "w") as f:
        json.dump(vars(args), f, indent=4)  # 保存模型信息
    logger.info(f"experiment finished, saved to: {exp_dir}")


if __name__ == "__main__":
    main()
