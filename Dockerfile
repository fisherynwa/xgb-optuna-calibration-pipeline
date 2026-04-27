# ─── Base image ───────────────────────────────────────────────────────────────
FROM python:3.12-slim

# ─── Install uv ───────────────────────────────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:0.7.0 /uv /usr/local/bin/uv

# ─── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ─── Copy dependency files first (layer caching) ─────────────────────────────
COPY pyproject.toml uv.lock ./

# ─── Install dependencies (exclude nvidia GPU packages) ───────────────────────
RUN uv sync --no-install-project \
    --no-install-package nvidia-nccl-cu12 \
    --no-install-package nvidia-cudnn-cu12 \
    --no-install-package nvidia-cublas-cu12 \
    --no-install-package nvidia-cuda-runtime-cu12 \
    --no-install-package nvidia-cuda-cupti-cu12 \
    --no-install-package nvidia-cuda-nvrtc-cu12 \
    --no-install-package nvidia-cufft-cu12 \
    --no-install-package nvidia-curand-cu12 \
    --no-install-package nvidia-cusolver-cu12 \
    --no-install-package nvidia-cusparse-cu12 \
    --no-install-package nvidia-nvjitlink-cu12 \
    --no-install-package nvidia-nvtx-cu12

# ─── Copy project files ───────────────────────────────────────────────────────
COPY src/        ./src/
COPY notebooks/  ./notebooks/
COPY main.py     ./

# ─── Install project ──────────────────────────────────────────────────────────
ENV PYTHONPATH=/app

# ─── Expose Jupyter port ──────────────────────────────────────────────────────
EXPOSE 8888
EXPOSE 5002
# ─── Launch Jupyter Lab ───────────────────────────────────────────────────────
CMD ["uv", "run", "jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--notebook-dir=/app", \
     "--ServerApp.token=''", \
     "--ServerApp.password=''"]