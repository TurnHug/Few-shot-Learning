import os
import json
import glob

import matplotlib.pyplot as plt  # 导入matplotlib用于可视化
import pandas as pd


def calculate_accuracy(df1, df2):

    # 根据'image_name'字段对两个数据框进行排序
    df1_sorted = df1.sort_values(by="img_name").reset_index(drop=True)
    df2_sorted = df2.sort_values(by="img_name").reset_index(drop=True)

    # 计算准确率
    correct_predictions = (df1_sorted["label"] == df2_sorted["label"]).sum()
    total_predictions = df1_sorted.shape[0]

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy


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
        plt.axis('off')  # 关闭坐标轴

    plt.tight_layout()
    return plt


def load_model_info(experiment_dir="experiments"):
    # 获取所有实验文件夹内的model_info.json文件
    model_info_files = glob.glob(os.path.join(experiment_dir, "*", "model_info.json"))
    
    all_model_info = []
    
    for file in model_info_files:
        with open(file, 'r') as f:
            model_info = json.load(f)
            all_model_info.append(model_info)
    
    # 将所有模型信息合并为一个DataFrame
    model_info_df = pd.DataFrame(all_model_info)
    
    # 打印显示DataFrame
    print(model_info_df)



if __name__ == "__main__":
    # 调用函数以加载和显示模型信息
    load_model_info()
