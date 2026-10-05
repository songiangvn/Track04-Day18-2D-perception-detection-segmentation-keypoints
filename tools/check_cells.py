"""Chạy nhanh các ô TODO của notebook trên CPU (không cần GPU) để xem phản hồi ✅/❌.
Dùng: python tools/check_cells.py <chỉ số ô> [...]   — luôn chạy trước ô import (4) và ô tiện ích (5).
Chỉ dùng cho ô không cần tải model; notebook đầy đủ chạy bằng slurm/run_notebook.sh."""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

nb = json.loads((Path(__file__).parents[1] / "lab_2d_perception_student.ipynb").read_text())
G = {"__name__": "__main__"}
for i in [4, 5] + [int(a) for a in sys.argv[1:]]:
    src = "".join(nb["cells"][i]["source"])
    src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
    print(f"── ô {i} ──")
    exec(compile(src, f"<cell {i}>", "exec"), G)
