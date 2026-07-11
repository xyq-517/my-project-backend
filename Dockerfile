# Dockerfile - 极简版（连 apt-get 都不需要）
# 解决 Railway 构建 OOM 问题：不执行任何 apt-get install
# 预生成模式只需要 Python + Flask + Pillow + numpy

FROM python:3.11-slim

WORKDIR /app

# 使用轻量版依赖（不含 torch、opencv）
COPY requirements-deploy.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 复制核心代码
COPY api_web_final.py .

# 复制数据文件（预生成图片、金标准、指标等）
COPY kits19_hash_map.json .
COPY per_image_metrics_kits19_dformer_EPA_test.csv .
COPY img_out/ ./img_out/
COPY VOCdevkit_kits19/ ./VOCdevkit_kits19/
COPY VOCdevkit_lidc_test/ ./VOCdevkit_lidc_test/
COPY miou_out/ ./miou_out/

# 暴露端口
EXPOSE 8000

# 启动命令
CMD gunicorn api_web_final:app --bind 0.0.0.0:${PORT:-8000} --workers 1 --timeout 120
