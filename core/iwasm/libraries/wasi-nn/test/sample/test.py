import numpy as np
import tensorflow as tf
from PIL import Image
import urllib.request
import os

# 下载图像

image_path = "dog_image.jpg"


# 加载标签文件（假设标签文件每行是一个类名）
def load_labels(label_file_path):
    with open(label_file_path, 'r') as f:
        labels = f.readlines()
    # 清除换行符
    labels = [label.strip() for label in labels]
    return labels

# 加载 TFLite 模型
interpreter = tf.lite.Interpreter(model_path="mobilenet_v2_1.0_224.tflite")
interpreter.allocate_tensors()

# 获取输入和输出张量的索引
input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()

# 加载并预处理图像
def load_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # 调整大小为 224x224
    img = np.array(img)  # 转换为 numpy 数组
    img = np.expand_dims(img, axis=0)  # 增加批量维度
    img = img.astype(np.float32)  # 转换为浮点型
    img = img / 255.0  # 归一化到 [0, 1]
    return img
def save_input_tensor_bin(input_data):
     # 将输入数据转换为 float32 类型并保存
    input_data = input_data.astype(np.float32)
    np.save('input_tensor_224.npy', input_data)

    print(input_data)
    print(input_data.shape)

    # 读取npy文件
    file_path = 'input_tensor_224.npy'
    data = np.load(file_path)

    # 输出文件的维度和一些样本数据
    print("文件的维度: ", data.shape)
    print("文件的第一个元素: ", data[0, 0, 0, 0])  # 打印第一个元素
    print("文件的第一个元素: ", data[0, 0, 0, 1])  # 打印第二个元素

    data.tofile('input_tensor_224.bin')
# 进行推理
def predict(image_path):
    img = load_image(image_path)
    
    # 设置输入张量
    interpreter.set_tensor(input_details[0]['index'], img)
    save_input_tensor_bin(img)
    
    # 执行推理
    interpreter.invoke()
    
    # 获取输出结果
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    return output_data

# 获取预测的标签
def get_predicted_label(output_data, labels):
    print("输出数据：", output_data)
    predicted_index = np.argmax(output_data)  # 获取最大概率的索引
    predicted_label = labels[predicted_index]  # 将索引映射到标签
    return predicted_label

# 测试函数
def test_model(image_path, label_file_path):
    # 加载标签文件
    labels = load_labels(label_file_path)
    
    # 进行预测
    output = predict(image_path)
    
    # 获取预测标签
    predicted_label = get_predicted_label(output, labels)
    
    print("预测结果：", predicted_label)
    return predicted_label

# 执行测试
label_file_path = "label_mobile.txt"  # 标签文件路径，确保该文件与代码在同一目录下
predicted_label = test_model(image_path, label_file_path)

# 清理下载的图像文件


