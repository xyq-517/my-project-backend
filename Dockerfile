FROM python:3.11-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 创建目录
RUN mkdir -p model_data logs

# 下载 ResNet50 预训练权重（替换成你的下载链接）
RUN wget -O model_data/resnet50-19c8e357.pth \
    "https://pth-1451618751.cos.ap-shanghai.myqcloud.com/model_data/resnet50-19c8e357.pth?q-sign-algorithm=sha1&q-ak=AKID3bnpzREsOdKbZ9rl_xSt_qQ3ZMbCXPpTv-SoZ0E5RUM5rAHTmP8FvqWthRuT-srz&q-sign-time=1783645497;1783649097&q-key-time=1783645497;1783649097&q-header-list=host&q-url-param-list=&q-signature=62c959b465cb9681029d9dbde26c1443caf8c31e&x-cos-security-token=c3ejoeCLtJHsFCjs9jv64VeehVXe3Lea6a80e0f25318b0679ea7f0de589b32de_jSo4YIAXWZNuytpdba63gi-P6mLyjINWyn5pCwUSWcV9PL_5lVBxNv-D677dYew1JOYgIFdOSQwoGLep2u1ByrUBRl3Xb68JH7LU0SzWFosb03m_2W0ibBeBgLPfLCn_VyK0iQrzKDytgE6ZuqXO556MxDFDTgmJG1wLs5b_myPZBjXIxagaXp2mlHCAc_a6_AdsbpP3dRuXRrb8eqCTVFY-Au4OcubKRMaU9a5b85ln0OJo8Co5oNhrKZ7kyKOkortyFAoMUbgfOjwDq1kWQ"

# 下载训练好的 UNet 模型（替换成你的下载链接）
RUN wget -O logs/best_epoch_weights.pth \
    "https://pth-1451618751.cos.ap-shanghai.myqcloud.com/best_epoch_weights.pth?q-sign-algorithm=sha1&q-ak=AKIDb0K8Svn2KY0FzfV_cr0_kT79UH-WxQWa8p0TgOUZ7s2LffGlDGXHWJRBiIdNDsw2&q-sign-time=1783645478;1783649078&q-key-time=1783645478;1783649078&q-header-list=host&q-url-param-list=&q-signature=d997091c099f19cca5c25feb93b23fb575c3cc4c&x-cos-security-token=c3ejoeCLtJHsFCjs9jv64VeehVXe3Lea97a775a5a4efd407dfcdb4f7371460e3_jSo4YIAXWZNuytpdba63s6-xa4__hkhZovEwMOWVPeuIx0dfbhHp-6nGee8l4Tu5L9wvWomY1lmeLmaiRYJGRGDie6j-Vsnrx8MUJc-M6WUMP8tyy-VUmWJweef0ptX41YE2V3kk4-iAw0_PWoneTAfaXYmGd96vlD1VWmjxFYeDhB36ErydN3JoFgzOMQk5RRxb4ONdXd6TNxbs97n74eEZKcoAj3UTv9o163wQPGRFr5QL2Ib_HLJ6JBhyh3IYVKa23373Bph66IQ5x7hDA"

COPY . .

EXPOSE 8000

CMD ["gunicorn", "api_web_final:app", "--bind", "0.0.0.0:8000"]
