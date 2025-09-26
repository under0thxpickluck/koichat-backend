# Dockerfile (RTX 5090 / Blackwell Compatible)
# ベースイメージをCUDA 12.8.1に変更
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# システムのタイムゾーンを設定
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 必要なツールをインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv git tini && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements.txtを先にコピーしてライブラリをインストール
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのソースコードをコピー
COPY . .

# ENTRYPOINTを設定
ENTRYPOINT ["/usr/bin/tini", "--"]

# ▼▼▼【ここを修正】FastAPIサーバーをUvicornで起動するように変更 ▼▼▼
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
