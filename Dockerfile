FROM python:3.11-slim

# 安装 scipy 和 opencv 所需的系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    gcc \
    gfortran \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# 先单独安装 scipy（避免依赖冲突）
RUN pip install --no-cache-dir numpy scipy

# 再安装其他依赖
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "api_web_final:app", "--bind", "0.0.0.0:8000"]