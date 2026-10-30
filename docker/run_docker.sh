#!/bin/bash

# Docker container launcher
# - Auto-detects GPU availability (falls back to CPU mode if no nvidia-smi)
# - Enables on-screen (XQuartz / SSH X11) rendering when DISPLAY is forwarded

set -e

TAG="latest"
while getopts "t:" opt; do
  case $opt in
    t) TAG="$OPTARG" ;;
  esac
done

IMAGE_NAME="robotics-safety:${TAG}"
CONTAINER_NAME="robotics-safety-container"

# Detect GPU availability
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected — running in GPU mode"
    GPU_FLAGS="--gpus all"
else
    echo "⚠ No GPU detected — running in CPU mode"
    GPU_FLAGS=""
fi

# X11 on-screen rendering setup
# - Only enable X11 flags when DISPLAY is set and xdpyinfo can reach the server
# - mujoco.viewer (GLFW→GLX) runs over SSH X11 via indirect GLX
# - To avoid "wrong authentication" errors when libX11 connects from inside the
#   container, we rewrite the cookie as FamilyWild (ffff) and bind-mount it.
X11_FLAGS=()
XAUTH_SRC="${XAUTHORITY:-$HOME/.Xauthority}"
if [[ -n "${DISPLAY:-}" ]] && command -v xdpyinfo &>/dev/null && xdpyinfo &>/dev/null; then
    if ! command -v xauth &>/dev/null; then
        echo "⚠ xauth not installed on host — cannot rewrite X11 cookie. Falling back to headless."
    elif [[ ! -f "$XAUTH_SRC" ]]; then
        echo "⚠ Xauthority file not found: $XAUTH_SRC — falling back to headless."
    else
        XAUTH_HOST="/tmp/xauth-robocasa-${USER}"
        : > "$XAUTH_HOST"
        XAUTHORITY="$XAUTH_SRC" xauth nlist "$DISPLAY" \
            | sed -e 's/^..../ffff/' \
            | xauth -f "$XAUTH_HOST" nmerge - 2>/dev/null
        if [[ -s "$XAUTH_HOST" ]]; then
            chmod 0644 "$XAUTH_HOST"
            echo "✓ X11 detected (DISPLAY=$DISPLAY) — on-screen rendering enabled"
            echo "  rewrote cookie as FamilyWild → $XAUTH_HOST"
            X11_FLAGS=(
                -e "DISPLAY=${DISPLAY}"
                -e "XAUTHORITY=/root/.Xauthority"
                -e "LIBGL_ALWAYS_INDIRECT=1"
                -e "__GLX_VENDOR_LIBRARY_NAME=mesa"
                -e "XDG_RUNTIME_DIR=/tmp/runtime-root"
                -v "${XAUTH_HOST}:/root/.Xauthority:ro"
                -v "/tmp/.X11-unix:/tmp/.X11-unix:rw"
            )
        else
            echo "⚠ no cookie for $DISPLAY in $XAUTH_SRC — falling back to headless."
        fi
    fi
else
    if [[ -n "${DISPLAY:-}" ]]; then
        echo "⚠ DISPLAY=${DISPLAY} is set but xdpyinfo failed — X11 forwarding is broken."
        echo "   If you need on-screen, reconnect with 'ssh -Y' and try again. Continuing headless."
    else
        echo "⚠ DISPLAY not set — running headless"
    fi
fi

# Stop the existing container if it's running
if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
    echo "Stopping existing container..."
    docker stop ${CONTAINER_NAME}
fi

# Remove the stopped container if it exists
if docker ps -aq -f name=${CONTAINER_NAME} | grep -q .; then
    echo "Removing existing container..."
    docker rm ${CONTAINER_NAME}
fi

# NVIDIA EGL vendor config — some Ubuntu 22.04 hosts don't ship
# /usr/share/glvnd/egl_vendor.d/10_nvidia.json, so we write it ourselves and
# bind-mount it as a single file into the container. Inside the container,
# mujoco/egl reads this JSON to find libEGL_nvidia.so.0.
EGL_VENDOR_DIR="${HOME}/.cache/robotics-safety-egl"
mkdir -p "${EGL_VENDOR_DIR}"
if [[ ! -f "${EGL_VENDOR_DIR}/10_nvidia.json" ]]; then
    cat > "${EGL_VENDOR_DIR}/10_nvidia.json" <<'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF
fi

echo "Starting Docker container..."

# Launch the container
docker run -it -d \
    --name ${CONTAINER_NAME} \
    ${GPU_FLAGS} \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e MUJOCO_GL=egl \
    -e PYOPENGL_PLATFORM=egl \
    -e __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    -v "${EGL_VENDOR_DIR}/10_nvidia.json:/usr/share/glvnd/egl_vendor.d/10_nvidia.json:ro" \
    -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
    -v $(pwd):/workspace \
    "${X11_FLAGS[@]}" \
    --workdir /workspace \
    --net=host \
    --privileged \
    --shm-size=8g \
    ${IMAGE_NAME}

# If on-screen mode is on, verify X11 connectivity from inside the container.
# xdpyinfo/xauth are diagnostic only — actual X11 forwarding relies on the
# bind-mounted /root/.Xauthority and the DISPLAY env var, so we keep going even
# if apt-get fails.
if [[ ${#X11_FLAGS[@]} -gt 0 ]]; then
    echo "Installing X11 utilities inside the container..."
    docker exec ${CONTAINER_NAME} bash -lc '
        mkdir -p /tmp/runtime-root && chmod 700 /tmp/runtime-root
        if ! command -v xdpyinfo >/dev/null; then
            apt-get update -qq >/dev/null 2>&1 || true
            apt-get install -y --no-install-recommends x11-utils xauth >/dev/null 2>&1 || true
        fi
    ' || true
    if docker exec ${CONTAINER_NAME} bash -c 'command -v xdpyinfo >/dev/null' 2>/dev/null; then
        if docker exec ${CONTAINER_NAME} xdpyinfo >/dev/null 2>&1; then
            echo "✓ Container can reach X server (${DISPLAY})"
        else
            echo "⚠ Container failed to reach X server (${DISPLAY}) — check your Mac XQuartz settings"
            echo "  (Preferences → Security → enable 'Allow connections from network clients',"
            echo "   then 'defaults write org.xquartz.X11 enable_iglx -bool true' and restart XQuartz)"
        fi
    else
        echo "⚠ xdpyinfo missing inside container (apt failed). Skipping diagnostics — failure will surface when running a demo."
    fi
fi

echo 'export XDG_RUNTIME_DIR=/tmp/runtime-root' >> ~/.bashrc
echo 'mkdir -p $XDG_RUNTIME_DIR 2>/dev/null' >> ~/.bashrc
source ~/.bashrc
docker exec -it ${CONTAINER_NAME} bash
