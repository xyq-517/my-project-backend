# api_web_final.py - 图像分割 API 服务
# 修复版：适配 Railway / Docker 部署

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

# ==================== 路径配置（修复：使用相对路径，兼容 Linux/Docker） ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HASH_MAP_FILE = os.path.join(BASE_DIR, "kits19_hash_map.json")
KITS19_GOLD_STANDARD_PATH = os.path.join(BASE_DIR, "VOCdevkit_kits19", "VOC2007", "SegmentationClass_Vis")
KITS19_PREDICT_PATH = os.path.join(BASE_DIR, "img_out")
LIDC_GOLD_STANDARD_PATH = os.path.join(BASE_DIR, "VOCdevkit_lidc_test", "VOC2007", "SegmentationClass")
LIDC_PREDICT_PATH = os.path.join(BASE_DIR, "img_out")
METRICS_PATH = os.path.join(BASE_DIR, "miou_out", "metrics.json")

# 每张图的分割系数CSV文件
KITS19_PER_IMAGE_CSV = os.path.join(BASE_DIR, "per_image_metrics_kits19_dformer_EPA_test.csv")
LIDC_PER_IMAGE_CSV = os.path.join(BASE_DIR, "miou_out", "per_image_metrics_LIDC_test.csv")

app = Flask(__name__)
CORS(app)

# 全局变量
unet = None
kits19_hash_map = {}
metrics_cache = {}
kits19_per_image_metrics = {}
lidc_per_image_metrics = {}


# ==================== 工具函数 ====================

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
                    metrics_dict[image_id] = {}
                    for key, value in row.items():
                        if key == 'image_id':
                            continue
                        try:
                            metrics_dict[image_id][key] = float(value)
                        except (ValueError, TypeError):
                            metrics_dict[image_id][key] = value
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


# ==================== API 路由 ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'kits19_count': len(kits19_hash_map),
        'kits19_csv_count': len(kits19_per_image_metrics),
        'lidc_csv_count': len(lidc_per_image_metrics)
    })


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


# ==================== 服务初始化（修复：模块级别初始化，gunicorn 也能执行） ====================

def initialize_service():
    """初始化服务：加载数据文件和模型"""
    global kits19_hash_map, metrics_cache, kits19_per_image_metrics, lidc_per_image_metrics, unet

    print("=" * 60)
    print("正在初始化 API 服务...")
    print(f"工作目录: {BASE_DIR}")

    # 加载哈希映射
    if os.path.exists(HASH_MAP_FILE):
        with open(HASH_MAP_FILE, 'r', encoding='utf-8') as f:
            kits19_hash_map = json.load(f)
        print(f"kits19 哈希映射: {len(kits19_hash_map)} 条")
    else:
        print(f"[警告] 哈希映射文件不存在: {HASH_MAP_FILE}")

    # 加载指标
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r', encoding='utf-8') as f:
            metrics_cache = json.load(f)
        print("指标数据已加载")
    else:
        print(f"[警告] 指标文件不存在: {METRICS_PATH}")

    # 加载每张图的分割系数CSV
    print("加载每张图的分割系数...")
    kits19_per_image_metrics = load_per_image_metrics(KITS19_PER_IMAGE_CSV)
    lidc_per_image_metrics = load_per_image_metrics(LIDC_PER_IMAGE_CSV)

    # 尝试加载 UNet 模型（备用，失败不影响预生成模式）
    print("尝试加载 UNet 模型（备用）...")
    try:
        from unet import Unet
        unet = Unet()
        print("UNet 模型加载完成")
    except Exception as e:
        print(f"[警告] UNet 模型加载失败（不影响预生成模式）: {e}")
        unet = None

    print("=" * 60)
    print("API 服务初始化完成")
    print(f"健康检查: GET /api/health")
    print(f"分割接口: POST /api/segment")
    print("=" * 60)


# 在模块加载时执行初始化（gunicorn 启动时也会执行）
initialize_service()


# ==================== 本地开发启动 ====================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5003))
    print(f"服务地址: http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
