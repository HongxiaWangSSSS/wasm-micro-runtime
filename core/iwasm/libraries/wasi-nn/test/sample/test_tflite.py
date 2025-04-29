import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time

# 配置路径
MODEL_PATH = './detection.tflite'    # 模型路径
LABELS_PATH = './labelmap.txt'       # 标签文件路径
IMAGE_PATH = './image.png'            # 输入图像路径
OUTPUT_PATH = './output.jpg'          # 输出检测结果路径

# 加载标签
def load_labels(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    labels = [line.strip() for line in lines if line.strip()]
    return labels

# 加载 tflite 模型
def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# 预处理图像
def preprocess_image(image_path, input_size):
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize(input_size, Image.Resampling.LANCZOS)  # 新版Pillow需要用Resampling
    input_data = np.array(img_resized, dtype=np.uint8)
    input_data = np.expand_dims(input_data, axis=0)  # 添加batch维度 (1, h, w, c)
    return input_data, img  # 返回输入数据和原始图像

# 执行检测
def detect_objects(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    # 将输入数据转换为 float32 类型并保存
    input_data = input_data.astype(np.uint8)
    np.save('input_tensor.npy', input_data)

    print(input_data)
    print(input_data.shape)

    # 读取npy文件
    file_path = 'input_tensor.npy'
    data = np.load(file_path)

    # 输出文件的维度和一些样本数据
    print("文件的维度: ", data.shape)
    print("文件的第一个元素: ", data[0, 0, 0, 0])  # 打印第一个元素
    print("文件的第一个元素: ", data[0, 0, 0, 1])  # 打印第二个元素

    data.tofile('input_tensor.bin')

    interpreter.invoke()

    # 获取输出
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]         # 位置
    classes = interpreter.get_tensor(output_details[1]['index'])[0]       # 类别
    scores = interpreter.get_tensor(output_details[2]['index'])[0]        # 置信度
    num = interpreter.get_tensor(output_details[3]['index'])[0]           # 检测数量

    results = []
    for i in range(int(num)):
        if scores[i] >= 0.5:  # 设置置信度阈值
            results.append({
                'bounding_box': boxes[i],
                'class_id': int(classes[i]),
                'score': scores[i]
            })
    return results

# 绘制检测结果
def draw_results(image, results, labels, output_path):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    width, height = image.size

    for obj in results:
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * width)
        xmax = int(xmax * width)
        ymin = int(ymin * height)
        ymax = int(ymax * height)

        class_id = obj['class_id']
        score = obj['score']
        label = labels[class_id] if class_id < len(labels) else "???"

        print(f"Detected {label} with confidence {score:.2f}")

        # 绘制矩形框
        draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=2)

        # 显示类别文字
        text = f"{label}: {score:.2f}"
        text_width=1
        text_height = 1
        # text_width, text_height = font.getsize(text)

        # 背景框
        draw.rectangle(
            [(xmin, ymin - text_height - 4), (xmin + text_width + 4, ymin)],
            fill='red'
        )

        # 类别文字
        draw.text((xmin + 2, ymin - text_height - 2), text, fill='white', font=font)

    image.save(output_path)
    print(f"结果已保存至：{output_path}")

# 主程序入口
if __name__ == "__main__":
    labels = load_labels(LABELS_PATH)
    interpreter = load_model(MODEL_PATH)

    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    input_size = (input_shape[2], input_shape[1])  # 注意顺序是 (宽, 高)

    input_data, original_image = preprocess_image(IMAGE_PATH, input_size)

    start_time = time.time()
    results = detect_objects(interpreter, input_data)
    end_time = time.time()

    print(f"推理耗时: {end_time - start_time:.2f}秒")

    draw_results(original_image, results, labels, OUTPUT_PATH)
