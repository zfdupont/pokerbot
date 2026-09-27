# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM --platform=linux/amd64 ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        build-essential \
        clang \
        lld \
        python3.12 \
        python3.12-dev \
        python3-pip \
        curl \
        zstd \
    && rm -rf /var/lib/apt/lists/*

# Buck2: GitHub prunes dated release artifacts (the 2026-06-30 tag matching the
# local ~/bin/buck2 now 404s), so pull the rolling `latest` release. The prelude
# ships bundled with the binary and .buckconfig uses external_cells=bundled, so a
# newer buck2 stays compatible with the BUCK rules.
RUN curl -fL \
    "https://github.com/facebook/buck2/releases/download/latest/buck2-x86_64-unknown-linux-gnu.zst" \
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
RUN python3.12 -c \
    "import sysconfig; p=sysconfig.get_path('include'); \
     open('.buckconfig.local','w').write(f'[python]\n  include_path = {p}\n[build]\n  cpu_tune = x86-64\n')"

# Copy sixmax source (separate layer so C++ changes don't bust the cache above)
COPY sixmax/ /build/sixmax/

# pybind11 headers: the repo's third_party/pybind11/include is a machine-local
# symlink (gitignored), so install pybind11 in-container to keep the build
# hermetic on any host (CI runner, native amd64).
RUN pip3 install --no-cache-dir --break-system-packages pybind11 \
    && PB_INC=$(python3.12 -c "import pybind11, os; print(os.path.join(os.path.dirname(pybind11.__file__), 'include'))") \
    && mkdir -p third_party/pybind11 \
    && ln -sfn "$PB_INC" third_party/pybind11/include

# Build the extension
RUN /usr/local/bin/buck2 build //sixmax:sixmax

# Extract the .so to a known path
RUN find /build/buck-out -name "sixmax.so" -exec cp {} /opt/sixmax.so \; \
    && test -f /opt/sixmax.so

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM --platform=linux/amd64 ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        python3.12 \
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
RUN uv pip install --system --python python3.12 \
        "numpy>=1.26" \
        "tomli>=2.0" \
        "tqdm>=4.66"

WORKDIR /pokerbot
