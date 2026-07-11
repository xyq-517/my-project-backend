#!/bin/bash
# start.sh - 启动脚本：从 COS 下载数据，然后启动 gunicorn
# 使用方法：在 Dockerfile 或 Procfile 中调用此脚本

set -e

echo "========================================="
echo "正在检查数据文件..."
echo "========================================="

DATA_DIR="/app"
cd "$DATA_DIR"

# ==================== 需要你配置的 COS 下载链接 ====================
# 把你的腾讯云 COS 下载链接填到下面的变量里
# 建议打包成 zip 文件，下载解压更快
# 公开桶直接用 URL，私有桶用带签名的临时链接（注意有效期）

# kits19 金标准图（VOCdevkit_kits19.zip）
KITS19_ZIP_URL="${KITS19_ZIP_URL:-}"

# LIDC 金标准图（VOCdevkit_lidc_test.zip）
LIDC_ZIP_URL="${LIDC_ZIP_URL:-}"

# 预生成预测图（img_out.zip）
IMG_OUT_ZIP_URL="${IMG_OUT_ZIP_URL:-}"

# 指标数据（miou_out.zip）
MIOU_OUT_ZIP_URL="${MIOU_OUT_ZIP_URL:-}"

# kits19 哈希映射
HASH_MAP_URL="${HASH_MAP_URL:-}"

# kits19 每张图指标 CSV
KITS19_CSV_URL="${KITS19_CSV_URL:-}"

# ==================================================================

# 函数：下载并解压 zip 文件
download_and_extract() {
    local url="$1"
    local dest_dir="$2"
    local zip_name="$3"

    if [ -z "$url" ]; then
        echo "[跳过] $zip_name 未配置 URL"
        return 0
    fi

    if [ -d "$dest_dir" ] && [ "$(ls -A $dest_dir 2>/dev/null)" ]; then
        echo "[已存在] $dest_dir 已有数据，跳过下载"
        return 0
    fi

    echo "[下载] $zip_name ..."
    if wget -q --timeout=60 -O "/tmp/$zip_name" "$url"; then
        echo "[解压] $zip_name ..."
        mkdir -p "$dest_dir"
        unzip -q "/tmp/$zip_name" -d "$DATA_DIR"
        rm "/tmp/$zip_name"
        echo "[完成] $zip_name 下载解压完成"
    else
        echo "[警告] $zip_name 下载失败，请检查 URL"
        return 1
    fi
}

# 函数：下载单个文件
download_file() {
    local url="$1"
    local dest_file="$2"
    local name="$3"

    if [ -z "$url" ]; then
        echo "[跳过] $name 未配置 URL"
        return 0
    fi

    if [ -f "$dest_file" ]; then
        echo "[已存在] $dest_file，跳过下载"
        return 0
    fi

    echo "[下载] $name ..."
    if wget -q --timeout=60 -O "$dest_file" "$url"; then
        echo "[完成] $name 下载完成"
    else
        echo "[警告] $name 下载失败"
        return 1
    fi
}

# 下载数据文件
echo ""
echo "--- 下载数据 ---"

download_and_extract "$KITS19_ZIP_URL" "VOCdevkit_kits19" "VOCdevkit_kits19.zip"
download_and_extract "$LIDC_ZIP_URL" "VOCdevkit_lidc_test" "VOCdevkit_lidc_test.zip"
download_and_extract "$IMG_OUT_ZIP_URL" "img_out" "img_out.zip"
download_and_extract "$MIOU_OUT_ZIP_URL" "miou_out" "miou_out.zip"
download_file "$HASH_MAP_URL" "kits19_hash_map.json" "kits19_hash_map.json"
download_file "$KITS19_CSV_URL" "per_image_metrics_kits19_dformer_EPA_test.csv" "kits19_per_image.csv"

echo ""
echo "========================================="
echo "数据准备完成，启动 API 服务..."
echo "========================================="
echo ""

# 启动 gunicorn
exec gunicorn api_web_final:app \
    --bind "0.0.0.0:${PORT:-8000}" \
    --workers 1 \
    --timeout 120 \
    --preload
