FROM python:3.11-slim

# 安装系统依赖（只需要一次）
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先安装 numpy+scipy 的预编译 wheel（从默认 PyPI）
RUN pip install --no-cache-dir --only-binary :all: numpy scipy==1.10.1

# 复制 requirements.txt 并安装其他依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "api_web_final:app", "--bind", "0.0.0.0:8000"]