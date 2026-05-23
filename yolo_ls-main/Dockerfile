# 使用 Ubuntu 基础镜像
FROM ubuntu:22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # PyQt5 依赖
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libdbus-1-3 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    # 视频编解码
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    # 清理
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖（使用国内镜像加速）
RUN pip3 install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    rm -rf /usr/local/lib/python3.10/dist-packages/cv2/qt

# 复制项目文件
COPY . .

# 创建数据目录
RUN mkdir -p /app/data /app/models

# 设置 Qt 环境变量（用于 GUI）
# 使用PyQt5的Qt插件，避免与OpenCV的Qt冲突
ENV QT_X11_NO_MITSHM=1
ENV DISPLAY=:0
ENV QT_QPA_PLATFORM_PLUGIN_PATH=/usr/local/lib/python3.10/dist-packages/PyQt5/Qt5/plugins/platforms
ENV QT_PLUGIN_PATH=/usr/local/lib/python3.10/dist-packages/PyQt5/Qt5/plugins

# 暴露端口
EXPOSE 8000

# 默认命令
CMD ["python3", "main.py", "--gui"]
