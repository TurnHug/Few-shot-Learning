# Few-shot-Learning
小样本学习之图像分类
### 训练模型

```bash
python train.py --train_dir dataset/train_set --model prototypical 

```

### 测试模型

```bash
python evaluate.py --test_dir dataset/test_set --true_labels_path dataset/test_set/query_labels.csv --test_models_dir experiments/2024-10-05_193151  

```

