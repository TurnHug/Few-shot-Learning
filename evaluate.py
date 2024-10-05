import argparse
import time
import logging
import os
import json
import glob
import torch
from tqdm import tqdm
from tabulate import tabulate
import matplotlib.pyplot as plt  # 导入matplotlib用于可视化
import pandas as pd

from dataset import get_dataloaders
import warnings

warnings.filterwarnings("ignore")


# 在文件开头添加以下代码来配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FewShotTester:
    def __init__(
        self,
        test_dataset,
        true_labels_path,
        test_model_dir,
        device="cuda",
    ):
        self.test_model_dir = test_model_dir
        self.test_dataset = test_dataset  # 测试数据集
        self.all_predictions = []  # 存储所有预测结果

        self.true_label_df = pd.read_csv(true_labels_path)  # 读取真实标签
        self.device = device
        self.model = self.load_model_info()  # 加载模型

    def load_model_info(self):
        model = torch.load(
            os.path.join(self.test_model_dir, "checkpoints", "best_model.pth"),
            map_location=self.device,
        )
        return model

    def evaluate(self, save_predictions=False, save_path=None):
        self.model.eval()  # 设置模型为评估模式
        total_correct = 0  # 记录总正确预测数
        total_predictions = 0  # 记录总预测数

        with torch.no_grad():
            for task_idx in tqdm(
                range(len(self.test_dataset.task_dirs)), desc="Predict ..."
            ):
                task_data = self.test_dataset.load_task(task_idx)  # 加载任务数据

                # 将数据移动到设备
                support_images = task_data["support_images"].to(self.device)
                support_labels = task_data["support_labels"].to(self.device)
                query_images = task_data["query_images"].to(self.device)

                # 进行预测
                predictions = self.model.predict(
                    support_images, support_labels, query_images, n_way=10
                )

                class_names = list(task_data["class_to_idx"].keys())  # 获取类别名称
                predicted_labels = [
                    class_names[pred] for pred in predictions.cpu().numpy()
                ]

                # 计算当前任务的准确率
                query_paths = task_data["query_paths"]  # 获取查询路径
                true_labels = self.true_label_df.set_index(
                    "img_name"
                )  # 将真实标签数据框设置为以文件名为索引
                correct_predictions = sum(
                    true_labels.loc[os.path.basename(query_paths[i]), "label"]
                    == predicted_labels[i]
                    for i in range(len(predicted_labels))
                    if os.path.basename(query_paths[i]) in true_labels.index
                )
                total_correct += correct_predictions
                total_predictions += len(predicted_labels)
                task_accuracy = (
                    correct_predictions / len(predicted_labels)
                    if len(predicted_labels) > 0
                    else 0
                )

                # 在tqdm中显示当前任务的准确率
                tqdm.write(
                    f"Task {task_idx + 1}/{len(self.test_dataset.task_dirs)} - Accuracy: {task_accuracy:.4f}"
                )

                # 如果需要保存预测结果
                if save_predictions:
                    for i, pred_label in enumerate(predicted_labels):
                        self.all_predictions.append(
                            {
                                "img_name": os.path.basename(
                                    task_data["query_paths"][i]
                                ),
                                "label": pred_label,
                            }
                        )
        predict_df = pd.DataFrame(self.all_predictions)
        # 保存预测结果到CSV文件
        if save_predictions and save_path:
            predict_df.to_csv(save_path, index=False)

        overall_accuracy = (
            total_correct / total_predictions if total_predictions > 0 else 0
        )
        logger.info(f"All Accuracy: {overall_accuracy:.4f}")
        return predict_df, overall_accuracy


def visualize_predictions(predictions_df, true_labels_df, num_images=9):

    # 合并数据框
    merged_df = pd.merge(
        predictions_df, true_labels_df, on="img_name", suffixes=("_pred", "_true")
    )

    # 随机选择要可视化的图像
    sample_df = merged_df.sample(n=num_images)

    # 创建可视化
    plt.figure(figsize=(10, 10))  # 调整图像大小
    for i, row in enumerate(sample_df.itertuples()):
        file_name = row.img_name
        parts = file_name.split("_")
        task_part = "_".join(parts[:2])  # 组合前两个部分
        img_path = os.path.join(
            "dataset/test_set/" + task_part + "/query/",
            file_name,
        )
        img = plt.imread(img_path)
        plt.subplot(3, 3, i + 1)  # 3行3列
        plt.imshow(img)
        plt.title(f"True: {row.label_true}\nPred: {row.label_pred}")
        plt.axis("off")  # 关闭坐标轴

    plt.tight_layout()
    return plt


def load_model_info(experiment_dir="experiments"):
    # 获取所有实验文件夹内的model_info.json文件
    model_info_files = glob.glob(os.path.join(experiment_dir, "*", "model_info.json"))

    all_model_info = []

    for file in model_info_files:
        with open(file, "r", encoding="utf-8") as f:
            model_info = json.load(f)
            all_model_info.append(model_info)

    # 将所有模型信息合并为一个DataFrame
    model_info_df = pd.DataFrame(all_model_info)

    show_df = model_info_df[
        [
            "model",
            "backbone_model",
            "learning_rate",
            "num_epochs",
            "accuracy",
            "test_time",
        ]
    ]

    return show_df


def parse_args():
    parser = argparse.ArgumentParser(description="小样本学习测试")
    parser.add_argument("--test_dir", type=str, required=True, help="测试数据集路径")
    parser.add_argument(
        "--true_labels_path", type=str, required=True, help="测试集真实标签路径"
    )
    parser.add_argument(
        "--test_models_dir",
        type=str,
        default="all",
        required=True,
        help="待测试模型保存路径",
    )

    parser.add_argument("--n_way", type=int, default=10, help="分类的类别数")
    parser.add_argument(
        "--k_shot", type=int, default=5, help="支持集中每个类别的样本数"
    )
    parser.add_argument(
        "--q_query", type=int, default=2, help="查询集中每个类别的样本数"
    )

    parser.add_argument(
        "--device", type=str, default="cuda", help="使用的设备 (cuda/cpu)"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(json.dumps(vars(args), indent=4))  # 记录参数信息
    since = time.time()
    test_dataset = get_dataloaders("", args.test_dir, "", is_train=False)

    tester = FewShotTester(
        test_dataset, args.true_labels_path, args.test_models_dir, args.device
    )
    predictions_df, accuracy = tester.evaluate(
        save_predictions=True,
        save_path=os.path.join(args.test_models_dir, "predictions.csv"),
    )

    plt = visualize_predictions(predictions_df, tester.true_label_df)  # 可视化预测结果
    plt.savefig(os.path.join(args.test_models_dir, "example_predict.png"))

    # 打开并读取模型信息文件
    with open(
        os.path.join(args.test_models_dir, "model_info.json"), "r", encoding="utf-8"
    ) as file:
        data = json.load(file)

    data["accuracy"] = round(accuracy, 4)
    data["test_time"] = round(time.time() - since, 3)
    # 将数据写入模型信息文件
    with open(
        os.path.join(args.test_models_dir, "model_info.json"), "w", encoding="utf-8"
    ) as file:
        json.dump(data, file, indent=4)

    infos = load_model_info()
    # 使用 tabulate 打印美观的表格
    print(tabulate(infos, headers="keys", tablefmt="pretty"))

if __name__ == "__main__":
    main()
