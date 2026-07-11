# data_loader.py - 数据下载与加载模块（纯 Python，零系统依赖）
# 功能：从 COS 下载 zip 数据并解压
# 可被 app.py / start.py / api_web_final.py 等任意入口调用

import os
import sys
import zipfile
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def download_file(url, dest_path, name="文件"):
    """下载文件，显示进度"""
    if os.path.exists(dest_path):
        print(f"[已存在] {name}: {dest_path}")
        return True

    if not url:
        print(f"[跳过] {name}: 未配置 URL")
        return False

    print(f"[下载] {name} ...")
    try:
        def report(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                mb = downloaded / 1024 / 1024
                total_mb = total_size / 1024 / 1024
                print(f"\r  进度: {percent:.1f}% ({mb:.1f}/{total_mb:.1f} MB)", end="")

        urllib.request.urlretrieve(url, dest_path, reporthook=report)
        print(f"\n[完成] {name} 下载完成")
        return True
    except Exception as e:
        print(f"\n[错误] {name} 下载失败: {e}")
        return False


def extract_zip(zip_path, extract_to, name="压缩包"):
    """解压 zip 文件"""
    if os.path.exists(extract_to) and os.listdir(extract_to):
        print(f"[已存在] {name} 已解压: {extract_to}")
        return True

    if not os.path.exists(zip_path):
        print(f"[跳过] {name}: zip 文件不存在")
        return False

    print(f"[解压] {name} ...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(BASE_DIR)
        print(f"[完成] {name} 解压完成")
        # 解压成功后删除 zip，节省空间
        try:
            os.remove(zip_path)
        except:
            pass
        return True
    except Exception as e:
        print(f"[错误] {name} 解压失败: {e}")
        return False


def download_all_data():
    """从环境变量读取 URL，下载并解压所有数据"""
    os.chdir(BASE_DIR)

    print("=" * 50)
    print("正在准备数据文件...")
    print("=" * 50)

    # 从环境变量获取 COS 下载链接
    kits19_zip_url = os.environ.get("KITS19_ZIP_URL", "")
    lidc_zip_url = os.environ.get("LIDC_ZIP_URL", "")
    img_out_zip_url = os.environ.get("IMG_OUT_ZIP_URL", "")
    miou_out_zip_url = os.environ.get("MIOU_OUT_ZIP_URL", "")
    hash_map_url = os.environ.get("HASH_MAP_URL", "")
    kits19_csv_url = os.environ.get("KITS19_CSV_URL", "")

    print(f"\n工作目录: {BASE_DIR}\n")

    # 下载并解压 kits19 金标准
    zip_path = os.path.join(BASE_DIR, "VOCdevkit_kits19.zip")
    if download_file(kits19_zip_url, zip_path, "VOCdevkit_kits19.zip"):
        extract_zip(zip_path, os.path.join(BASE_DIR, "VOCdevkit_kits19"), "VOCdevkit_kits19")

    # 下载并解压 LIDC 金标准
    zip_path = os.path.join(BASE_DIR, "VOCdevkit_lidc_test.zip")
    if download_file(lidc_zip_url, zip_path, "VOCdevkit_lidc_test.zip"):
        extract_zip(zip_path, os.path.join(BASE_DIR, "VOCdevkit_lidc_test"), "VOCdevkit_lidc_test")

    # 下载并解压预生成预测图
    zip_path = os.path.join(BASE_DIR, "img_out.zip")
    if download_file(img_out_zip_url, zip_path, "img_out.zip"):
        extract_zip(zip_path, os.path.join(BASE_DIR, "img_out"), "img_out")

    # 下载并解压指标数据
    zip_path = os.path.join(BASE_DIR, "miou_out.zip")
    if download_file(miou_out_zip_url, zip_path, "miou_out.zip"):
        extract_zip(zip_path, os.path.join(BASE_DIR, "miou_out"), "miou_out")

    # 下载单个文件
    if hash_map_url:
        download_file(hash_map_url, os.path.join(BASE_DIR, "kits19_hash_map.json"), "kits19_hash_map.json")

    if kits19_csv_url:
        download_file(kits19_csv_url, os.path.join(BASE_DIR, "per_image_metrics_kits19_dformer_EPA_test.csv"), "kits19_per_image.csv")

    print("\n" + "=" * 50)
    print("数据准备完成")
    print("=" * 50 + "\n")

    return True


if __name__ == "__main__":
    download_all_data()
