#!/bin/bash
# Usage: pym path/to/file.py
# Converts to: python -m path.to.file (without .py)

script_path="$1"

if [[ ! -f "$script_path" ]]; then
    echo "❌ File does not exist: $script_path"
    exit 1
fi

# 去掉 .py 后缀
module_path="${script_path%.py}"

# 替换路径中的 / 为 . （适配 Windows 的 \ 也处理一下）
module_path="${module_path//\//.}"
module_path="${module_path//\\/.}"

# 执行 python -m xxx
echo "▶️ Running: python -m $module_path"
python -m "$module_path"
