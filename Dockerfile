# Dockerfile - COS 数据版（镜像超小，构建超快）
# 数据文件从腾讯云 COS 下载，不打包进镜像
# 镜像体积：~200MB，构建时间：~1分钟

FROM python:3.11-slim

# 只安装 wget 和 unzip（用于下载解压数据）
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 安装 Python 依赖
COPY requirements-deploy.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码文件（只有代码，没有数据！）
COPY api_web_final.py .
COPY start.sh .

# 给启动脚本执行权限
RUN chmod +x start.sh

# 暴露端口
EXPOSE 8000

# 启动：先下载数据，再启动服务
CMD ["./start.sh"]
