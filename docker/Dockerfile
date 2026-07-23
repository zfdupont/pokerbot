# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM --platform=linux/amd64 ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        build-essential \
        python3.10 \
        python3.10-dev \
        curl \
        zstd \
    && rm -rf /var/lib/apt/lists/*

# Pin Buck2 to match local version (~/bin/buck2 --version = 2026-06-30)
RUN curl -fL \
    "https://github.com/facebook/buck2/releases/download/2026-06-30/buck2-x86_64-unknown-linux-gnu.zst" \
    | zstd -d - -o /usr/local/bin/buck2 \
    && chmod +x /usr/local/bin/buck2

WORKDIR /build

# Copy Buck2 config and build infrastructure (cached layers — changes rarely)
COPY .buckconfig /build/
COPY toolchains/ /build/toolchains/
COPY third_party/ /build/third_party/
COPY common/ /build/common/
COPY BUCK /build/

# Set python include path and cpu_tune for cross-compile (x86-64, not native)
RUN python3.10 -c \
    "import sysconfig; p=sysconfig.get_path('include'); \
     open('.buckconfig.local','w').write(f'[python]\n  include_path = {p}\n[build]\n  cpu_tune = x86-64\n')"

# Copy sixmax source (separate layer so C++ changes don't bust the cache above)
COPY sixmax/ /build/sixmax/

# Build the extension
RUN /usr/local/bin/buck2 build //sixmax:sixmax

# Extract the .so to a known path
RUN find /build/buck-out -name "sixmax.so" -exec cp {} /opt/sixmax.so \; \
    && test -f /opt/sixmax.so

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM --platform=linux/amd64 ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        python3.10 \
        python3.10-distutils \
        curl \
        rsync \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -fLsS https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy compiled extension
COPY --from=builder /opt/sixmax.so /opt/sixmax.so
ENV SIXMAX_SO_PATH=/opt/sixmax.so

# Install Python deps (system-wide — no venv needed in container)
COPY pyproject.toml /tmp/pyproject.toml
RUN uv pip install --system \
        "numpy>=1.26" \
        "tomli>=2.0" \
        "tqdm>=4.66"

WORKDIR /pokerbot
