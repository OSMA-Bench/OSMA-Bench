# Dockerfile for RoboCasa/RoboSuite Safety Benchmark
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    MUJOCO_GL=egl \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# 필수 시스템 패키지 설치 및 Python 3.10 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    # 기본 도구
    git curl wget unzip ca-certificates \
    # 빌드 도구
    build-essential cmake ninja-build \
    # Python 3.10 및 pip
    python3.10 python3.10-dev python3.10-distutils python3-pip \
    # OpenGL/EGL 라이브러리
    libgl1-mesa-glx libegl1-mesa libgles2-mesa-dev \
    libglvnd0 libglfw3-dev libglew-dev \
    # X11 라이브러리 (headless GUI 지원) + on-screen (SSH X11/XQuartz) 진단 도구
    libx11-6 libxext6 libxrender1 libxi6 libxrandr2 libxkbcommon0 \
    x11-utils xauth \
    # 멀티미디어
    libglib2.0-0 ffmpeg \
    # 기타 유틸리티
    htop vim nano \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean \
 && python3.10 -m pip install --no-cache-dir imageio-ffmpeg

# python3, python, pip3, pip 모두 python3.10으로 심볼릭 링크
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Python 기본 설정
RUN python3 -m pip install --upgrade pip setuptools wheel

# 워크스페이스 생성
WORKDIR /workspace

# 소스 코드 복사
COPY . /workspace/

# git submodule 초기화 및 업데이트
# RUN git submodule update --init --recursive
RUN git submodule update --init && \
    git submodule foreach --quiet 'git submodule update --init 2>/dev/null || true'


# Voxposer 설치
WORKDIR /workspace/policy/Voxposer
RUN pip install -r requirements.txt

# RoboSuite 설치
WORKDIR /workspace/benchmark/robosuite
RUN pip install -e .

# RoboCasa 설치
WORKDIR /workspace/benchmark/robocasa
RUN pip install -e .
RUN pip install pre-commit; pre-commit install 

# Vllm 설치
RUN pip install 'vllm==0.11.2'
RUN pip install 'numpy==1.26.4'
# 워크스페이스로 돌아가기
WORKDIR /workspace

# 기본 실행 명령어
CMD ["/bin/bash"]
