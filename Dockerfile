# Dockerfile - 修复版：适配 Railway 部署
# 修复内容：
# 1. 添加 OpenCV 系统依赖（libgl1, libglib2.0-0 等）
# 2. 复制数据文件（img_out, VOCdevkit, miou_out, json, csv）
# 3. 使用 $PORT 环境变量（Railway 自动注入）
# 4. 预生成模式不需要下载模型（节省构建时间和镜像体积）

FROM python:3.11-slim

# 安装系统依赖（OpenCV 需要 libgl1, libglib2.0-0 等）
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先复制 requirements.txt 并安装依赖（利用 Docker 缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 创建必要目录
RUN mkdir -p model_data logs img_out miou_out

# 复制核心代码
COPY api_web_final.py .
COPY unet.py .
COPY nets/ ./nets/
COPY utils/ ./utils/

# 复制数据文件（预生成图片、金标准、指标等）
# 注意：确保这些文件在 GitHub 仓库中存在
COPY kits19_hash_map.json .
COPY per_image_metrics_kits19_dformer_EPA_test.csv .

# 复制预生成预测图
COPY img_out/ ./img_out/

# 复制 kits19 金标准图
COPY VOCdevkit_kits19/ ./VOCdevkit_kits19/

# 复制 LIDC 金标准图
COPY VOCdevkit_lidc_test/ ./VOCdevkit_lidc_test/

# 复制指标数据
COPY miou_out/ ./miou_out/

# 暴露端口（Railway 会通过 $PORT 环境变量覆盖）
EXPOSE 8000

# 启动命令：使用 $PORT 环境变量，兼容 Railway
CMD gunicorn api_web_final:app --bind 0.0.0.0:${PORT:-8000} --workers 1 --timeout 120
