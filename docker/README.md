# Docker Setup for Alpamayo-R1

## ビルド方法

```bash
cd docker
docker-compose build
```

## 実行方法

```bash
docker-compose run alpamayo
```

## コンテナ内での作業

コンテナ内に入ったら:

```bash
# HuggingFace認証
hf auth login

# 推論テストの実行
python src/alpamayo_r1/test_inference.py
```

## 注意事項

- GPUを使用するため、NVIDIA Container Toolkit が必要です
- ROS2 Humble がプリインストールされています
- Python 3.12 がデフォルトのPythonバージョンです
