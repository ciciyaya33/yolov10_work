import os
import cv2
import xml.etree.ElementTree as ET
import time  # 导入time模块
import onnxruntime as ort
import numpy as np

# 创建 xml_results 文件夹（如果不存在）
results_folder = 'xml_results'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# 指定要处理的图片文件夹
input_folder = 'datat'  # 替换为你的图片文件夹路径
onnx_model_path = "best.onnx"

# 加载 ONNX 模型
session = ort.InferenceSession(onnx_model_path)

# 获取模型输入输出信息
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 遍历输入文件夹中的所有图片
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  # 你可以根据需要添加其他格式
        img_path = os.path.join(input_folder, filename)

        # 记录开始时间
        start_time = time.time()

        # 读取图片并进行预处理（假设输入尺寸为 640x640）
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (640, 640))  # 调整大小为模型所需尺寸
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # 转换为RGB
        img_normalized = img_rgb / 255.0  # 归一化
        img_input = np.transpose(img_normalized, (2, 0, 1))  # 转换为 (C, H, W)
        img_input = np.expand_dims(img_input, axis=0)  # 添加 batch 维度

        # 进行预测
        outputs = session.run([output_name], {input_name: img_input.astype(np.float32)})

        # 记录结束时间
        end_time = time.time()

        # 计算预测时间
        prediction_time = end_time - start_time

        # 创建 XML 文件
        root = ET.Element("annotations")

        # 遍历检测结果并填充 XML
        detections = outputs[0]  # 假设输出是 (batch_size, num_boxes, 6) 的形状
        for detection in detections[0]:  # 遍历检测到的每个目标
            bbox = detection[:4]  # 获取边界框坐标 (xmin, ymin, xmax, ymax)
            confidence = detection[4]  # 获取置信度
            class_id = int(detection[5])  # 获取类标识

            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = str(class_id)  # 使用类标识作为名称
            ET.SubElement(obj, "confidence").text = str(confidence)
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(bbox[0]))
            ET.SubElement(bndbox, "ymin").text = str(int(bbox[1]))
            ET.SubElement(bndbox, "xmax").text = str(int(bbox[2]))
            ET.SubElement(bndbox, "ymax").text = str(int(bbox[3])) + "\n"

        # 保存 XML 文件
        xml_filename = os.path.splitext(filename)[0] + '.xml'
        xml_path = os.path.join(results_folder, xml_filename)
        tree = ET.ElementTree(root)
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)

        # 打印预测结果和时间
        print(f"Results for {filename} saved to {xml_path}")
        print(f"Prediction time for {filename}: {prediction_time:.4f} seconds")
