FROM python:3.11-slim

# 安装 scipy 所需的系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    gfortran \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先只复制 requirements.txt
COPY requirements.txt .

# 安装所有依赖（因为设置了 PIP_PREFER_BINARY，pip会优先下载预编译包）
RUN pip install --no-cache-dir -r requirements.txt

# 再复制其余代码
COPY . .

EXPOSE 8000

CMD ["gunicorn", "api_web_final:app", "--bind", "0.0.0.0:8000"]