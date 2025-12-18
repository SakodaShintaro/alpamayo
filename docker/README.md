# Docker Setup for Alpamayo-R1

## ビルド方法

ホストユーザーと同じUID/GIDでビルド：

```bash
cd docker
USER_ID=$(id -u) GROUP_ID=$(id -g) USERNAME=$(whoami) docker-compose build
```

## 実行方法

```bash
USER_ID=$(id -u) GROUP_ID=$(id -g) docker-compose run alpamayo
```

## コンテナ内での作業

コンテナ内に入ったら:

```bash
# HuggingFace認証
hf auth login

# 推論テストの実行
python3 src/alpamayo_r1/test_inference.py
```

## autowareビルド

```bash
cd ~
git clone https://github.com/autowarefoundation/autoware
cd autoware/src
git clone https://github.com/ros-geographic-info/geographic_info.git -b jazzy
cd ../
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TOOL=ON --packages-up-to autoware_msgs

# 全ビルド
sudo apt update
export PIP_BREAK_SYSTEM_PACKAGES=1
rosdep install -y --from-paths src --ignore-src --rosdistro jazzy --skip-keys python3-torch
```

## 注意事項

- GPUを使用するため、NVIDIA Container Toolkit が必要です
- ROS2 Jazzy がプリインストールされています（Ubuntu 24.04ベース）
- Python 3.12 がデフォルトのPythonバージョンです
- ROS2 bag の互換性: Humble と Jazzy 間でbagファイルの互換性があります
- コンテナはホストユーザーと同じUID/GIDで実行されます
- `sudo`コマンドはパスワードなしで使用できます（必要に応じて）
