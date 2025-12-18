FROM osrf/ros:jazzy-desktop-full

# Install CUDA 12.1 (required for flash-attn)
RUN apt-get update && apt-get install -y wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-6 && \
    rm cuda-keyring_1.1-1_all.deb && \
    rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda-12.6
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install build essentials
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set up workspace
WORKDIR /workspace

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY src/ ./src/

# Install project dependencies to system Python (not in venv)
RUN uv pip install --system --break-system-packages -e .

# Install sudo
RUN apt-get update && apt-get install -y sudo && rm -rf /var/lib/apt/lists/*

# Create user with same UID/GID as host (will be set at runtime)
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=user

# Remove existing user/group with same UID/GID if exists, then create new user
RUN (getent passwd ${USER_ID} && userdel -r $(getent passwd ${USER_ID} | cut -d: -f1)) || true && \
    (getent group ${GROUP_ID} && groupdel $(getent group ${GROUP_ID} | cut -d: -f1)) || true && \
    groupadd -g ${GROUP_ID} ${USERNAME} && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Setup for both root and user
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc && \
    echo "source /opt/ros/jazzy/setup.bash" >> /home/${USERNAME}/.bashrc && \
    echo "export PATH=/root/.local/bin:\$PATH" >> /home/${USERNAME}/.bashrc

# Switch to user
USER ${USERNAME}
WORKDIR /workspace

CMD ["/bin/bash"]
