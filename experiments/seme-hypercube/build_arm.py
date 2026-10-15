#!/usr/bin/env python3
"""臂构建器：run_template.py + vendor/ → run.py。

Kaggle kernels push 只执行 code_file、不携带子目录文件
（1014 实测：B0 首版 ModuleNotFoundError: No module named 'src'），
故 vendor 代码以 b64 + sha256 自校验内嵌进 run.py。

正本：run_template.py（逻辑）+ vendor/（原仓库 src 与本仓
seme_graph / seme_adapter 的 vendored 副本）。
派生：run.py（部署形态，本构建器生成，可随时重生成对表）。

用法: python3 build_arm.py <臂目录>     # 如 python3 build_arm.py B0
"""
import base64
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARM = (HERE / sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE

BASE_VENDOR_FILES = [
    "src/__init__.py",
    "src/config.py",
    "src/data.py",
    "src/graph.py",
    "src/model.py",
    "src/metrics.py",
    "src/train.py",
    "src/deepwalk.py",
    "seme_graph.py",
    "seme_adapter.py",
]

# 臂自有模块（如 D0 的 directional.py）：vendor/ 根下
# 除基础清单外的 .py 自动追加（build_arm 扩展，
# 界面实验臂引入新模块所需）。
extra = sorted(
    p.relative_to(ARM / "vendor").as_posix()
    for p in (ARM / "vendor").glob("*.py")
    if p.relative_to(ARM / "vendor").as_posix()
    not in BASE_VENDOR_FILES
)
VENDOR_FILES = BASE_VENDOR_FILES + extra

vendor = {}
for rel in VENDOR_FILES:
    raw = (ARM / "vendor" / rel).read_bytes()
    vendor[rel] = [base64.b64encode(raw).decode(),
                   hashlib.sha256(raw).hexdigest()]

template = (ARM / "run_template.py").read_text()
marker = "_VENDOR = {}  # BUILD_INJECT_VENDOR"
if marker not in template:
    sys.exit(f"build_arm: {ARM / 'run_template.py'} 缺注入标记")
inject = "_VENDOR = " + json.dumps(vendor, ensure_ascii=False)
out = template.replace(marker, inject)
(ARM / "run.py").write_text(out)
print(f"built {ARM / 'run.py'}: {len(out)} bytes, "
      f"vendor {len(vendor)} files embedded")
