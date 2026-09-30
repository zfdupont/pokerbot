# syntax=docker/dockerfile:1
# ── Stage 1: build sixmax.so (mirrors docker/Dockerfile) ─────────────────────
FROM --platform=linux/amd64 ubuntu:24.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        build-essential clang lld python3.12 python3.12-dev python3-pip \
        curl zstd \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fL \
    "https://github.com/facebook/buck2/releases/download/latest/buck2-x86_64-unknown-linux-gnu.zst" \
    | zstd -d - -o /usr/local/bin/buck2 && chmod +x /usr/local/bin/buck2
WORKDIR /build
COPY .buckconfig /build/
COPY toolchains/ /build/toolchains/
COPY third_party/ /build/third_party/
COPY common/ /build/common/
COPY BUCK /build/
RUN python3.12 -c \
    "import sysconfig; p=sysconfig.get_path('include'); \
     open('.buckconfig.local','w').write(f'[python]\n  include_path = {p}\n[build]\n  cpu_tune = x86-64\n')"
COPY sixmax/ sixmax/
RUN pip3 install --no-cache-dir --break-system-packages pybind11 \
    && PB_INC=$(python3.12 -c "import pybind11,os;print(os.path.join(os.path.dirname(pybind11.__file__),'include'))") \
    && mkdir -p third_party/pybind11 && ln -sfn "$PB_INC" third_party/pybind11/include
# libtorch is excluded (.dockerignore) but third_party/BUCK coerces its targets;
# create non-linked placeholders so package loading succeeds.
RUN mkdir -p third_party/libtorch/lib third_party/libtorch/include/torch/csrc/api/include \
    && touch third_party/libtorch/lib/libc10.dylib \
             third_party/libtorch/lib/libtorch_cpu.dylib \
             third_party/libtorch/lib/libtorch.dylib
RUN /usr/local/bin/buck2 build //sixmax:sixmax \
    && find /build/buck-out -name sixmax.so -exec cp {} /opt/sixmax.so \; \
    && test -f /opt/sixmax.so

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM --platform=linux/amd64 python:3.12-slim
ENV PYTHONUNBUFFERED=1 SIXMAX_SO_PATH=/opt/sixmax.so
COPY --from=builder /opt/sixmax.so /opt/sixmax.so
COPY web/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /pokerbot
COPY web/ web/
COPY agents/ agents/
COPY game/ game/
COPY models/ models/
COPY util/ util/
COPY sixmax/ sixmax/
COPY cfr/ cfr/
EXPOSE 8100
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8100/health').status==200 else 1)"
CMD ["python", "-m", "web.main", "--host", "0.0.0.0", "--port", "8100"]
