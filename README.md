# Few-shot-Learning
小样本学习之图像分类
### 训练模型

```bash
python train.py --train_dir dataset/train_set --model prototypical --backbone_model resnet18 --learning_rate 0.01 --num_epochs 50 --n_way 10 --k_shot 5 --q_query 2 --device cuda

```

### 测试模型

```bash
python evaluate.py --test_dir dataset/test_set --true_labels_path dataset/test_set/query_labels.csv --test_models_dir experiments/2024-10-05_193151  --n_way 10 --k_shot 5 --q_query 2 --device cuda

```
