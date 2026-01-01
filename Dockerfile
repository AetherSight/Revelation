FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装Poetry
RUN pip install --no-cache-dir poetry

# 配置Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# 复制Poetry配置文件
COPY pyproject.toml poetry.lock* ./

# 安装Python依赖
RUN poetry install --no-dev && rm -rf $POETRY_CACHE_DIR

# 复制应用代码
COPY src/ ./src/

# 创建models目录（用于挂载）
RUN mkdir -p /app/models

# 设置环境变量
ENV MODEL_DIR=/app/models \
    PORT=5000 \
    PYTHONPATH=/app

# 暴露端口
EXPOSE 5000

# 启动应用
CMD ["poetry", "run", "python", "-m", "revelation"]

