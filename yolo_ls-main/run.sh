#!/bin/bash
# Docker 运行脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== 实时交通分析系统 Docker 启动脚本 ===${NC}"

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}错误: Docker 未安装${NC}"
    exit 1
fi

# 检查 NVIDIA Docker
if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
    echo -e "${YELLOW}警告: NVIDIA Docker 运行时未检测到，GPU 加速可能不可用${NC}"
fi

# 允许 X11 连接（GUI 模式需要）
if [ -n "$DISPLAY" ]; then
    xhost +local:docker 2>/dev/null || true
fi

# 创建必要目录
mkdir -p data models videos

# 解析参数
MODE="gui"
VIDEO_SOURCE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --cli)
            MODE="cli"
            shift
            ;;
        --source)
            VIDEO_SOURCE="$2"
            shift 2
            ;;
        --build)
            echo -e "${GREEN}构建 Docker 镜像...${NC}"
            docker-compose build
            exit 0
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --gui         启动 GUI 模式（默认）"
            echo "  --cli         启动命令行模式"
            echo "  --source FILE 指定视频源"
            echo "  --build       仅构建镜像"
            echo "  --help        显示帮助"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# 运行容器
if [ "$MODE" = "gui" ]; then
    echo -e "${GREEN}启动 GUI 模式...${NC}"
    docker-compose up traffic-analysis
else
    echo -e "${GREEN}启动命令行模式...${NC}"
    if [ -n "$VIDEO_SOURCE" ]; then
        docker-compose run --rm traffic-analysis-cli \
            python3 main.py --source "$VIDEO_SOURCE" --headless
    else
        docker-compose --profile cli up traffic-analysis-cli
    fi
fi
