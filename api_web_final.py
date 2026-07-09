# api_web_final.py - 图像分割 API 服务

import sys
import os
import io
import json
import base64
import hashlib
import time
import csv

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
def download_model():
    model_path = "model_data/vgg16-397923af.pth"
    if not os.path.exists(model_path):
        print("📥 正在下载模型权重...")
        os.makedirs("model_data", exist_ok=True)
        # 把你复制的临时链接粘贴到下面的引号里
        url = "https://pth-1451618751.cos.ap-shanghai.myqcloud.com/model_data/vgg16-397923af.pth?q-sign-algorithm=sha1&q-ak=AKIDaF8lsYAomTEhSBB9JFV1yKL0S6wJ85ZTfkqIQiAqsldg7uQi5XMcTkvLy0wQKueh&q-sign-time=1783577892;1783581492&q-key-time=1783577892;1783581492&q-header-list=host&q-url-param-list=&q-signature=1ace12468da824411593537fc3bb901bb570636e&x-cos-security-token=XyLqfqgTpDYGzmq50rve7YSbEeHch1da0f9e85684b9a38cdf746303c0915b2f8tPMyQ5GAEyZeDgv3OdZ9N32gUDQxQhqfBgqMk88NNMmZVVEn2aHSob6-C4k4vm41QA7QCLO3rldnXKF_HZvOBZEE-13C-8asyTL13HmyqFfy5s-8kulszEDcIj0il6DmSBjQScaddj9i46Pbzm2GjJCYKnKnT8DjtDhEVVUYoGNMzQhLqHdcjP9_2vdAMZkpIVPRmJ1ezw7JHGhGvYitXSv7SW9RdW10CT1I-Dhwy-IgJrk0s9aUVOP2wC6e-YfixTYpwvA9r9R7JIUy6GCHGQ"
        urllib.request.urlretrieve(url, model_path)
        print("✅ 模型下载完成！")
import os
import sys

# 强制设置 UTF-8 编码，避免字符集问题
if sys.platform == 'win32':
    import locale
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
# 配置
HASH_MAP_FILE = r"D:\shengwu\using\unet-pytorch\kits19_hash_map.json"
KITS19_GOLD_STANDARD_PATH = r"D:\shengwu\using\unet-pytorch\VOCdevkit_kits19\VOC2007\SegmentationClass_Vis"
KITS19_PREDICT_PATH = r"D:\shengwu\using\unet-pytorch\img_out"
LIDC_GOLD_STANDARD_PATH = r"D:\shengwu\using\unet-pytorch\VOCdevkit_lidc_test\VOC2007\SegmentationClass"
LIDC_PREDICT_PATH = r"D:\shengwu\using\unet-pytorch\img_out"
METRICS_PATH = r"D:\shengwu\using\unet-pytorch\miou_out\metrics.json"

# 每张图的分割系数CSV文件
KITS19_PER_IMAGE_CSV = r"D:\shengwu\using\unet-pytorch\per_image_metrics_kits19_dformer_EPA_test.csv"
LIDC_PER_IMAGE_CSV = r"D:\shengwu\using\unet-pytorch\miou_out\per_image_metrics_LIDC_test.csv"

app = Flask(__name__)
CORS(app)

# 全局变量
unet = None
kits19_hash_map = {}
metrics_cache = {}
kits19_per_image_metrics = {}
lidc_per_image_metrics = {}


def image_to_base64(image):
    """将 PIL Image 转换为 base64 字符串"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def load_image_to_base64(file_path):
    """加载图片文件并转换为 base64"""
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    return None


def load_per_image_metrics(csv_path):
    """加载每张图的分割系数CSV文件（包含所有指标列）"""
    metrics_dict = {}
    if not os.path.exists(csv_path):
        print(f"[警告] CSV文件不存在: {csv_path}")
        return metrics_dict

    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row.get('image_id', '')
                if image_id:
                    # 加载所有列（除 image_id 外）
                    metrics_dict[image_id] = {}
                    for key, value in row.items():
                        if key == 'image_id':
                            continue
                        try:
                            metrics_dict[image_id][key] = float(value)
                        except (ValueError, TypeError):
                            metrics_dict[image_id][key] = value  # 保留原始值（如 'nan'）
        print(f"[加载] {csv_path}: {len(metrics_dict)} 条记录")
    except Exception as e:
        print(f"[错误] 加载CSV失败: {e}")

    return metrics_dict


def get_metrics_from_csv(image_name, dataset):
    """从CSV文件中获取指定图片的分割系数"""
    base_name = image_name.replace('.jpg', '').replace('.png', '')
    if dataset == 'kits19':
        return kits19_per_image_metrics.get(base_name, {})
    elif dataset == 'lidc':
        return lidc_per_image_metrics.get(base_name, {})
    return {}


def clean_nan_value(value):
    """清理NaN值，将Python的NaN转换为JSON兼容的值"""
    if isinstance(value, float) and (value != value):  # NaN的判断: NaN != NaN
        return '-'
    return value


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({'status': 'ok', 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')})


@app.route('/api/segment', methods=['POST'])
def segment_image():
    """图像分割 API - 两个数据集都用预生成图片"""
    start_time = time.time()

    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '未上传文件'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '文件名不能为空'}), 400

        filename = file.filename
        file_content = file.stream.read()
        image_hash = hashlib.md5(file_content).hexdigest()

        print(f"\n[请求] 图像分割: {filename}")
        print(f"[哈希] {image_hash}")

        # kits19 数据集
        if image_hash in kits19_hash_map:
            print("[匹配] kits19 数据集")
            match_info = kits19_hash_map[image_hash]
            original_name = match_info['original_name']

            # 预测图路径（jpg格式）
            predict_name = original_name.replace('.jpg', '.jpg')  # 保持原样
            predict_path = os.path.join(KITS19_PREDICT_PATH, predict_name)

            # 金标准路径（png格式）
            gold_name = original_name.replace('.jpg', '.png')
            gold_path = os.path.join(KITS19_GOLD_STANDARD_PATH, gold_name)

            print(f"[查找] 预测图: {predict_path}")
            print(f"[查找] 金标准: {gold_path}")

            # 加载预测图
            if os.path.exists(predict_path):
                segment_base64 = load_image_to_base64(predict_path)
                print(f"[成功] 加载预测图: {predict_name}")
            else:
                print(f"[警告] 预测图不存在: {predict_path}")
                segment_base64 = None

            # 加载金标准
            if os.path.exists(gold_path):
                gold_base64 = load_image_to_base64(gold_path)
                print(f"[成功] 加载金标准: {gold_name}")
            else:
                print(f"[警告] 金标准不存在: {gold_path}")
                gold_base64 = None

            # 从CSV获取分割系数
            csv_metrics = get_metrics_from_csv(original_name, 'kits19')
            print(f"[系数] CSV匹配: {csv_metrics}")

            # 构建返回的分割系数（整体指标）
            metrics = {
                'accuracy': clean_nan_value(csv_metrics.get('accuracy', '-')),
                'dice': clean_nan_value(csv_metrics.get('mDice', '-')),
                'iou': clean_nan_value(csv_metrics.get('mIoU', '-')),
                'hd95': clean_nan_value(csv_metrics.get('mHD95', '-')),
                'precision': clean_nan_value(csv_metrics.get('mPrecision', '-')),
                'recall': clean_nan_value(csv_metrics.get('mRecall', '-')),
                'HD95_kidney': clean_nan_value(csv_metrics.get('HD95_kidney', '-'))
            }

            # 各类别详细指标
            per_class_metrics = {}
            for key, value in csv_metrics.items():
                if key.startswith('IoU_') or key.startswith('Dice_') or key.startswith('HD95_'):
                    per_class_metrics[key] = clean_nan_value(value)
            metrics['per_class'] = per_class_metrics

            return jsonify({
                'success': True,
                'dataset': 'kits19',
                'method': '预生成',
                'original_name': original_name,
                'predict_name': predict_name,
                'gold_name': gold_name,
                'image_hash': image_hash,
                'processing_time': round(time.time() - start_time, 2),
                'images': {
                    'original': base64.b64encode(file_content).decode('utf-8'),
                    'segment': segment_base64,
                    'gold_standard': gold_base64
                },
                'metrics': metrics
            })

        # LIDC 数据集 - 也用预生成图片
        else:
            print("[匹配] LIDC 数据集")

            # LIDC 使用上传的文件名查找预生成图片
            base_name = filename.replace('.jpg', '').replace('.png', '')

            # 预测图路径（jpg格式）
            predict_name = base_name + '.jpg'
            predict_path = os.path.join(LIDC_PREDICT_PATH, predict_name)

            # 金标准路径（png格式）
            gold_name = base_name + '.png'
            gold_path = os.path.join(LIDC_GOLD_STANDARD_PATH, gold_name)

            print(f"[查找] 预测图: {predict_path}")
            print(f"[查找] 金标准: {gold_path}")

            # 加载预测图
            if os.path.exists(predict_path):
                segment_base64 = load_image_to_base64(predict_path)
                print(f"[成功] 加载预测图: {predict_name}")
            else:
                print(f"[警告] 预测图不存在: {predict_path}")
                segment_base64 = None

            # 加载金标准
            if os.path.exists(gold_path):
                gold_base64 = load_image_to_base64(gold_path)
                print(f"[成功] 加载金标准: {gold_name}")
            else:
                print(f"[警告] 金标准不存在: {gold_path}")
                gold_base64 = None

            # 从CSV获取分割系数
            csv_metrics = get_metrics_from_csv(filename, 'lidc')
            print(f"[系数] CSV匹配: {csv_metrics}")

            # 构建返回的分割系数（整体指标）
            metrics = {
                'accuracy': clean_nan_value(csv_metrics.get('accuracy', '-')),
                'dice': clean_nan_value(csv_metrics.get('mDice', '-')),
                'iou': clean_nan_value(csv_metrics.get('mIoU', '-')),
                'hd95': clean_nan_value(csv_metrics.get('mHD95', '-')),
                'precision': clean_nan_value(csv_metrics.get('mPrecision', '-')),
                'recall': clean_nan_value(csv_metrics.get('mRecall', '-')),
                'HD95_kidney': clean_nan_value(csv_metrics.get('HD95_kidney', '-'))
            }

            # 各类别详细指标
            per_class_metrics = {}
            for key, value in csv_metrics.items():
                if key.startswith('IoU_') or key.startswith('Dice_') or key.startswith('HD95_'):
                    per_class_metrics[key] = clean_nan_value(value)
            metrics['per_class'] = per_class_metrics

            return jsonify({
                'success': True,
                'dataset': 'lidc',
                'method': '预生成',
                'original_name': filename,
                'predict_name': predict_name,
                'gold_name': gold_name,
                'image_hash': image_hash,
                'processing_time': round(time.time() - start_time, 2),
                'images': {
                    'original': base64.b64encode(file_content).decode('utf-8'),
                    'segment': segment_base64,
                    'gold_standard': gold_base64
                },
                'metrics': metrics
            })

    except Exception as e:
        print(f"[错误] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("正在初始化 API 服务...")

    # 加载哈希映射
    if os.path.exists(HASH_MAP_FILE):
        with open(HASH_MAP_FILE, 'r', encoding='utf-8') as f:
            kits19_hash_map = json.load(f)
        print(f"kits19 哈希映射: {len(kits19_hash_map)} 条")

    # 加载指标
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r', encoding='utf-8') as f:
            metrics_cache = json.load(f)
        print("指标数据已加载")

    # 加载每张图的分割系数CSV
    print("加载每张图的分割系数...")
    kits19_per_image_metrics = load_per_image_metrics(KITS19_PER_IMAGE_CSV)
    lidc_per_image_metrics = load_per_image_metrics(LIDC_PER_IMAGE_CSV)

    # 加载模型（备用）
    print("加载 UNet 模型...")
    from unet import Unet

    unet = Unet()
    print("UNet 模型加载完成")

    print("=" * 60)
    print("服务地址: http://0.0.0.0:5003")
    print("API 端点: POST /api/segment")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5003, debug=False, threaded=True)
