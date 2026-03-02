#!/bin/bash

set -e  # 出现错误时终止脚本

# === 配置路径 ===
# === 读取输入参数 ===
if [ $# -lt 1 ]; then
    echo "Usage: $0 <DATA_DIR>"
    exit 1
fi

DATA_DIR="$1"   # 从命令行第一个参数读取


CROPFOR_CONFIG="configs/entityv2/entity_segmentation/cropformer_hornet_3x.yaml"
CROPFOR_WEIGHTS="ckpts/CropFormer_hornet_3x_03823a.pth"

echo "[INFO] Running CropFormer module..."
python run_cropformer.py \
    --config-file "$CROPFOR_CONFIG" \
    --scene_dir "$DATA_DIR" \
    --image_path_pattern "images/*" \
    --dataset scannet \
    --opts MODEL.WEIGHTS "$CROPFOR_WEIGHTS"