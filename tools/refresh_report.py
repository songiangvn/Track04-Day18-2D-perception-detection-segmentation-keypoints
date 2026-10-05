"""Sau mỗi lần chạy notebook: thay bảng ONNX trong submission/bao_cao_bonus.md bằng số trong submission/bonus_onnx_latency.json."""
import json, re
from pathlib import Path

sub = Path(__file__).parents[1] / "submission"
d = json.loads((sub / "bonus_onnx_latency.json").read_text(encoding="utf-8"))
head = "| Ảnh | Cấu hình | preprocess (ms) | ONNX (ms) | postprocess (ms) | tổng (ms) | ứng viên qua conf | số box |\n|---|---|---:|---:|---:|---:|---:|---:|\n"
short = lambda s: "cảnh đông 3×3" if s.startswith("cảnh đông") else s
rows = "".join(f"| {short(r['Ảnh'])} | {r['Cấu hình']} | {r['preprocess (ms)']:.2f} | {r['ONNX inference (ms)']:.2f} | {r['postprocess (ms)']:.2f} "
               f"| {r['tổng (ms)']:.2f} | {r['ứng viên qua conf']} | {r['số box']} |\n" for r in d["rows"])
p = sub / "bao_cao_bonus.md"
txt, n = re.subn(r"\| Ảnh \| Cấu hình \|.*?\n(?:\|.*\n)+", lambda _: head + rows, p.read_text(encoding="utf-8"), count=1)
assert n == 1, "không tìm thấy bảng ONNX trong báo cáo"
p.write_text(txt, encoding="utf-8")
col = lambda k, c: [r[c] for r in d["rows"] if r["Cấu hình"].startswith(k)]
rng = lambda v: f"{min(v):.2f}–{max(v):.2f}"
print(f"cập nhật bảng ({d['cpu']}, {d['threads']} luồng). postprocess o2m {rng(col('one-to-many', 'postprocess (ms)'))} ms, "
      f"o2o {rng(col('one-to-one', 'postprocess (ms)'))} ms; tổng o2m {rng(col('one-to-many', 'tổng (ms)'))}, o2o {rng(col('one-to-one', 'tổng (ms)'))} ms")
