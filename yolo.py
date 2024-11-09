import os
from ultralytics import YOLO

# 设置参数
default_yaml_path = "/root/autodl-tmp/yolov10-main/ultralytics/cfg/default.yaml"  # default.yaml的路径
data_yaml_path = "/root/autodl-tmp/yolov10-main/datasets/bvm/data.yaml"           # data.yaml的路径
model_path = "/root/autodl-tmp/yolov10-main/yolov10n.pt"                          # 预训练模型路径（或自定义模型路径）

# 检查文件路径
if not os.path.exists(data_yaml_path):
    print(f"data.yaml 配置文件未找到：{data_yaml_path}")

# 初始化模型并加载配置
model = YOLO(model_path)  # 加载预训练模型

# 开始训练并明确指定数据集配置
model.train(data=data_yaml_path)  # 显式传递data.yaml文件路径
