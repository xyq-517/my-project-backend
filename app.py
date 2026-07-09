"""
肾脏肿瘤分割 Flask 后端服务
参考 predict.py 实现真实推理
"""

import os
import uuid
import time
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np

# 导入你的 Unet 模型类（参考 predict.py）
from unet import Unet
# 导入指标计算函数
from utils.utils_metrics import fast_hist, per_class_iu, per_class_PA_Recall, per_class_Precision, per_Accuracy, compute_single_hd95


app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置
UPLOAD_FOLDER = 'uploads'          # 上传图片保存目录
RESULT_FOLDER = 'results'          # 分割结果保存目录
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

# 创建目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 参考 predict.py 的方式初始化模型
print("正在加载模型...")
unet = Unet()
print("模型加载完成！")

# 参考 predict.py 的参数设置
count = True
# 肾脏肿瘤分割类别：背景、肾脏、肿瘤
name_classes = ["background", "kidney", "tumor"]
num_classes = len(name_classes)

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def compute_dice(pred, label, class_id):
    """计算单个类别的 Dice 系数"""
    pred_mask = (pred == class_id).astype(np.float32)
    label_mask = (label == class_id).astype(np.float32)
    intersection = np.sum(pred_mask * label_mask)
    union = np.sum(pred_mask) + np.sum(label_mask)
    if union == 0:
        return 0.0 if np.sum(label_mask) == 0 else 0.0
    return (2. * intersection) / (union + 1e-5)

@app.route('/api/segment', methods=['POST'])
def segment_image():
    """
    肾脏肿瘤分割接口
    接收上传的CT图片，返回分割结果和指标
    """
    try:
        # 检查是否有文件
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "message": "没有上传文件"
            }), 400

        file = request.files['file']
        
        # 检查是否上传了标签文件（用于计算指标）
        label_file = request.files.get('label')

        # 检查文件名
        if file.filename == '':
            return jsonify({
                "success": False,
                "message": "文件名为空"
            }), 400

        # 检查文件类型
        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "message": "不支持的文件格式，请上传 png, jpg, jpeg, bmp, tif, tiff 格式的图片"
            }), 400

        # 生成唯一文件名
        file_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        original_filename = f"{timestamp}_{file_id}_original.{file_ext}"
        result_filename = f"{timestamp}_{file_id}_result.png"

        # 保存上传的文件
        original_path = os.path.join(UPLOAD_FOLDER, original_filename)
        file.save(original_path)
        print(f"保存上传文件: {original_path}")

        # 加载图片并进行分割（参考 predict.py 的方式）
        print("开始分割...")
        start_time = time.time()

        # 使用 PIL 打开图片（与 predict.py 相同）
        image = Image.open(original_path)
        image_width, image_height = image.size

        # 参考 predict.py 的方式调用 detect_image
        r_image = unet.detect_image(image, count=count, name_classes=name_classes)

        # 如果返回的是元组（count=True时），提取图像和像素计数
        class_pixel_count = {}
        if isinstance(r_image, tuple):
            r_image, class_pixel_count = r_image

        # 保存分割结果
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        r_image.save(result_path)
        print(f"保存分割结果: {result_path}")

        # 计算处理时间
        process_time = time.time() - start_time

        # 初始化指标字典
        metrics = {
            "accuracy": 0.0,
            "mDice": 0.0,
            "mIoU": 0.0,
            "mPrecision": 0.0,
            "mRecall": 0.0,
            "Dice_kidney": 0.0,
            "IoU_kidney": 0.0,
            "Dice_tumor": 0.0,
            "IoU_tumor": 0.0,
            "HD95_kidney": None,
            "HD95_tumor": None,
            "image_width": image_width,
            "image_height": image_height
        }

        # 如果提供了标签文件，计算所有指标
        if label_file and allowed_file(label_file.filename):
            label_filename = f"{timestamp}_{file_id}_label.png"
            label_path = os.path.join(UPLOAD_FOLDER, label_filename)
            label_file.save(label_path)
            print(f"保存标签文件: {label_path}")
            
            # 读取标签和预测结果
            gt_mask = np.array(Image.open(label_path).convert('L'))
            pred_mask = np.array(r_image.convert('L'))
            
            # 确保尺寸一致
            if pred_mask.shape != gt_mask.shape:
                pred_mask = np.array(Image.fromarray(pred_mask).resize(gt_mask.shape[::-1]))
            
            # 计算混淆矩阵
            hist = fast_hist(gt_mask.flatten(), pred_mask.flatten(), num_classes)
            
            # 计算各类指标
            IoUs = per_class_iu(hist)
            PA_Recall = per_class_PA_Recall(hist)
            Precision = per_class_Precision(hist)
            accuracy = per_Accuracy(hist)
            
            # 计算各类别 Dice
            Dice_kidney = compute_dice(pred_mask, gt_mask, 1)  # kidney 类别索引为1
            Dice_tumor = compute_dice(pred_mask, gt_mask, 2)   # tumor 类别索引为2
            
            # 计算 HD95
            kidney_pred = (pred_mask == 1).astype(np.float32)
            kidney_gt = (gt_mask == 1).astype(np.float32)
            tumor_pred = (pred_mask == 2).astype(np.float32)
            tumor_gt = (gt_mask == 2).astype(np.float32)
            
            if np.sum(kidney_pred) > 0 and np.sum(kidney_gt) > 0:
                metrics["HD95_kidney"] = round(float(compute_single_hd95(kidney_pred, kidney_gt)), 4)
            elif np.sum(kidney_pred) == 0 and np.sum(kidney_gt) == 0:
                metrics["HD95_kidney"] = 0.0
            else:
                metrics["HD95_kidney"] = round(float(np.sqrt(image_width**2 + image_height**2)), 4)
            
            if np.sum(tumor_pred) > 0 and np.sum(tumor_gt) > 0:
                metrics["HD95_tumor"] = round(float(compute_single_hd95(tumor_pred, tumor_gt)), 4)
            elif np.sum(tumor_pred) == 0 and np.sum(tumor_gt) == 0:
                metrics["HD95_tumor"] = 0.0
            else:
                metrics["HD95_tumor"] = round(float(np.sqrt(image_width**2 + image_height**2)), 4)
            
            # 填充指标
            metrics["accuracy"] = round(float(accuracy), 4)
            metrics["mIoU"] = round(float(np.nanmean(IoUs)), 4)
            metrics["mRecall"] = round(float(np.nanmean(PA_Recall)), 4)
            metrics["mPrecision"] = round(float(np.nanmean(Precision)), 4)
            metrics["Dice_kidney"] = round(float(Dice_kidney), 4)
            metrics["IoU_kidney"] = round(float(IoUs[1]) if len(IoUs) > 1 else 0.0, 4)
            metrics["Dice_tumor"] = round(float(Dice_tumor), 4)
            metrics["IoU_tumor"] = round(float(IoUs[2]) if len(IoUs) > 2 else 0.0, 4)
            
            # 计算 mDice（平均 Dice）
            dice_scores = [Dice_kidney, Dice_tumor]
            metrics["mDice"] = round(float(np.mean([d for d in dice_scores if d > 0]) if any(d > 0 for d in dice_scores) else 0), 4)

        # 构建返回结果
        response_data = {
            "success": True,
            "message": "分割成功",
            "data": {
                "original_image": f"/uploads/{original_filename}",
                "segmented_image": f"/results/{result_filename}",
                "process_time_seconds": round(process_time, 3),
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics
            }
        }

        # 在后端打印返回的内容
        print("\n" + "="*80)
        print("分割结果返回内容:")
        print("="*80)
        print(f"文件名: {file.filename}")
        print(f"图片尺寸: {image_width} x {image_height}")
        print(f"处理时间: {process_time:.3f}秒")
        print("\n分割指标:")
        print(f"  accuracy: {metrics['accuracy']}")
        print(f"  mDice: {metrics['mDice']}")
        print(f"  mIoU: {metrics['mIoU']}")
        print(f"  mPrecision: {metrics['mPrecision']}")
        print(f"  mRecall: {metrics['mRecall']}")
        print(f"  Dice_kidney: {metrics['Dice_kidney']}")
        print(f"  IoU_kidney: {metrics['IoU_kidney']}")
        print(f"  Dice_tumor: {metrics['Dice_tumor']}")
        print(f"  IoU_tumor: {metrics['IoU_tumor']}")
        print(f"  HD95_kidney: {metrics['HD95_kidney']}")
        print(f"  HD95_tumor: {metrics['HD95_tumor']}")
        print("="*80 + "\n")

        return jsonify(response_data)

    except Exception as e:
        print(f"分割出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"分割失败: {str(e)}"
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "success": True,
        "message": "服务正常运行",
        "model_loaded": unet is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/uploads/<filename>')
def serve_upload(filename):
    """提供上传文件的访问"""
    from flask import send_from_directory
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/results/<filename>')
def serve_result(filename):
    """提供结果文件的访问"""
    from flask import send_from_directory
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == '__main__':
    print("=" * 50)
    print("肺结节分割服务启动")
    print(f"上传目录: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"结果目录: {os.path.abspath(RESULT_FOLDER)}")
    print("=" * 50)

    port = int(os.environ.get("PORT", 5000))
    # 启动服务，允许外部访问
    app.run(host='0.0.0.0', port=port, debug=False)

