# Dockerfile - 超轻量版（预生成模式专用）
# 修复：解决 Railway 构建时 OOM（内存不足）问题
# 说明：预生成模式不需要 PyTorch / OpenCV，镜像体积从几GB降到几百MB

FROM python:3.11-slim

# 只安装最基本的系统依赖（预生成模式不需要 libgl、build-essential 等）
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 使用轻量版依赖（不含 torch、opencv 等大体积库）
COPY requirements-deploy.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 复制核心代码
COPY api_web_final.py .

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
# --workers 1 防止内存不足，--timeout 120 防止大请求超时
CMD gunicorn api_web_final:app --bind 0.0.0.0:${PORT:-8000} --workers 1 --timeout 120
