import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 强制使用 CPU
from PIL import Image
from tqdm import tqdm

from unet import Unet
from utils.utils_metrics import compute_mIoU_and_Dice, show_results

# 注释掉损坏图片检查代码，加快运行速度
# for root, _, files in os.walk(r"D:\shengwu\using\unet-pytorch\VOCdevkit_kits19"):
#     for f in files:
#         path = os.path.join(root, f)
#         try:
#             with Image.open(path) as img:
#                 img.load()
#         except Exception as e:
#             print(f"损坏: {path}")

# img_dir = r"D:\shengwu\using\unet-pytorch\VOCdevkit_kits19\VOC2007\JPEGImages"
# count = 0
# for f in os.listdir(img_dir):
#     if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
#         continue
#     path = os.path.join(img_dir, f)
#     try:
#         with Image.open(path) as img:
#             img.load()
#     except Exception as e:
#         os.remove(path)
#         count += 1
#         print(f"已删除损坏图片: {path}")
# print(f"\n共删除 {count} 张损坏图片")
'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照JPG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
3、仅有按照VOC格式数据训练的模型可以利用这个文件进行miou的计算。
'''
if __name__ == "__main__":
    miou_mode = 0
    num_classes = 3
    name_classes = ["background", "kidney", "tumor"]
    VOCdevkit_path = r'D:\shengwu\using\unet-pytorch\VOCdevkit_kits19'

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/test.txt"),'r').read().splitlines() 
    gt_dir = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path = "miou_out"
    pred_dir = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        print("Load model.")
        unet = Unet()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image = Image.open(image_path)
            image = unet.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou and Dice.")
        hist, IoUs, PA_Recall, Precision, Dice = compute_mIoU_and_Dice(gt_dir, pred_dir, image_ids, num_classes, name_classes)
        print("Get miou and Dice done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, Dice, name_classes)

        # =============================================
        # 保存指标到 JSON 文件，供 api_server.py 读取
        # =============================================
        metrics = {
            "num_classes": num_classes,
            "name_classes": name_classes,
            "mDice": float(round(Dice.mean() * 100, 2)),
            "mIoU": float(round(IoUs.mean() * 100, 2)),
            "mPrecision": float(round(Precision.mean() * 100, 2)),
            "mRecall": float(round(PA_Recall.mean() * 100, 2)),
            "mPA": float(round(PA_Recall.mean() * 100, 2)),  # PA = Recall
            "per_class": {}
        }

        # 逐类别保存
        for i, name in enumerate(name_classes):
            metrics["per_class"][name] = {
                "Dice": float(round(Dice[i] * 100, 2)),
                "IoU": float(round(IoUs[i] * 100, 2)),
                "Precision": float(round(Precision[i] * 100, 2)),
                "Recall": float(round(PA_Recall[i] * 100, 2))
            }

        # 保存到 miou_out/metrics.json
        metrics_path = os.path.join(miou_out_path, "metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        print(f"保存指标到 {metrics_path}")
        print(f"  mDice: {metrics['mDice']}%")
        print(f"  mIoU: {metrics['mIoU']}%")
        print(f"  mPrecision: {metrics['mPrecision']}%")
        print(f"  mRecall: {metrics['mRecall']}%")

