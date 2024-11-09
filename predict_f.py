import os
import cv2
import xml.etree.ElementTree as ET
from ultralytics import YOLOv10

# 创建 xml_results 文件夹（如果不存在）
results_folder = 'xml_results'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# 指定要处理的图片文件夹
input_folder = 'datat'  # 替换为你的图片文件夹路径
model = YOLOv10("best.pt")

# 遍历输入文件夹中的所有图片
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  # 你可以根据需要添加其他格式
        img_path = os.path.join(input_folder, filename)
        
        # 进行预测
        results = model.predict(img_path)

        # 创建 XML 文件
        root = ET.Element("annotations")
        
        # 遍历检测结果并填充 XML
        for detection in results[0].boxes:
            bbox = detection.xyxy[0]  # 获取边界框坐标
            confidence = detection.conf[0]  # 获取置信度
            class_id = int(detection.cls[0])  # 获取类标识
            
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = str(class_id)  # 使用类标识作为名称
            ET.SubElement(obj, "confidence").text = str(confidence)
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(bbox[0]))
            ET.SubElement(bndbox, "ymin").text = str(int(bbox[1]))
            ET.SubElement(bndbox, "xmax").text = str(int(bbox[2]))
            ET.SubElement(bndbox, "ymax").text = str(int(bbox[3]))

        # 保存 XML 文件
        xml_filename = os.path.splitext(filename)[0] + '.xml'
        xml_path = os.path.join(results_folder, xml_filename)
        tree = ET.ElementTree(root)
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)

        print(f"Results for {filename} saved to {xml_path}")
