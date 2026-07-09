FROM python:3.11-slim

# 安装基础系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    gfortran \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先单独安装 scipy（从预编译的 wheel 安装）
RUN pip install --no-cache-dir \
    --only-binary :all: \
    --extra-index-url https://www.piwheels.org/simple \
    scipy==1.10.1

# 复制 requirements.txt 并安装其他依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "api_web_final:app", "--bind", "0.0.0.0:8000"]