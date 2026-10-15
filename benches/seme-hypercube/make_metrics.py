#!/usr/bin/env python3
"""聚合器：report_structure.json（正本）→ metrics.json（派生）。

纯标准库；check_gates.py 的输入。训练期指标接入后，本聚合器
按 prereg.md §5 扩展（第二批门限流程已在 prereg 冻结）。

用法:
    python3 make_metrics.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def main():
    report = json.loads((HERE / "report_structure.json").read_text())
    metrics = {
        "bench": report["bench"],
        "source_reports": ["report_structure.json"],
        **report["invariants"],
    }
    (HERE / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
