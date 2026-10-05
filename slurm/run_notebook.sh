#!/bin/bash
# Chạy toàn bộ notebook trên một GPU (tương đương "Restart session and run all" trên Colab).
#   Trên biomedia-slurm:  sbatch /vol/biomedic3/gn425/LabInAction/Lab03/slurm/run_notebook.sh
# Notebook được chạy trên một bản chụp lúc job bắt đầu (kể cả khi có ô lỗi):
#   • bản đã chạy luôn lưu ở Lab03/slurm_logs/executed_<jobid>.ipynb;
#   • chỉ khi KHÔNG có ô nào lỗi và notebook gốc không bị sửa trong lúc job chạy,
#     bản đã chạy mới được chép đè lên lab_2d_perception_student.ipynb (bản để nộp).
# Log: Lab03/slurm_logs/

#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic3/gn425/LabInAction/Lab03/slurm_logs/%x.%j.%N.log
#SBATCH --time=0-03:00:00
#SBATCH --job-name=lab03-nb

REPO=/vol/biomedic3/gn425/LabInAction/Lab03
NB=lab_2d_perception_student.ipynb
export PYTHONNOUSERSITE=1                                   # không lẫn gói trong ~/.local
export TORCH_HOME=/vol/biomedic3/gn425/.cache/torch         # weights torchvision, tránh đầy quota /homes
export YOLO_CONFIG_DIR=$REPO/.ultralytics                   # settings riêng cho lab, không đụng ~/.config/Ultralytics
export YOLO_AUTOINSTALL=False                               # cấm ultralytics tự pip install vào env chung (từng ghi đè onnxruntime-gpu)
export MPLBACKEND=module://matplotlib_inline.backend_inline

source /vol/biomedic3/gn425/miniconda3/bin/activate /vol/biomedic3/gn425/envs/labinaction
cd "$REPO" && mkdir -p slurm_logs "$YOLO_CONFIG_DIR"         # thư mục phải có sẵn, không thì ultralytics lùi về /tmp
echo "host=$(hostname) job=$SLURM_JOB_ID"
python -c "import torch, ultralytics; print(torch.__version__, ultralytics.__version__, torch.cuda.get_device_name(0))"
# dataset tiger-pose tải về Lab03/datasets (đã .gitignore); tắt gửi telemetry
python -c "from ultralytics import settings; settings.update(datasets_dir='$REPO/datasets', weights_dir='$REPO/weights', runs_dir='$REPO/runs', sync=False)"

SNAP=.run_${SLURM_JOB_ID}.ipynb                               # nằm cạnh notebook gốc → kernel có cwd = Lab03
OUT=slurm_logs/executed_${SLURM_JOB_ID}.ipynb
cp "$NB" "$SNAP"
jupyter nbconvert --to notebook --execute --allow-errors --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 "$SNAP" --output "$OUT" --output-dir .
rm -f "$SNAP"

python - "$NB" "$OUT" <<'PY'
import json, shutil, sys
nb, out = sys.argv[1:]
src = lambda p: [c["source"] for c in json.load(open(p))["cells"]]
cells = json.load(open(out))["cells"]
errs = [(i, o.get("ename"), o.get("evalue", "")[:200]) for i, c in enumerate(cells) if c["cell_type"] == "code"
        for o in c.get("outputs", []) if o["output_type"] == "error"]
for i, name, val in errs:
    print(f"❌ ô {i}: {name}: {val}")
if errs:
    print(f"Có {len(errs)} ô lỗi → KHÔNG chép đè {nb}. Xem {out}.")
elif src(nb) != src(out):
    print(f"⚠️  {nb} đã bị sửa trong lúc job chạy → KHÔNG chép đè. Bản đã chạy: {out}.")
else:
    shutil.copy(out, nb)
    print(f"✅ Chạy hết, không ô nào lỗi → đã cập nhật {nb} (kèm output) và submission/.")
PY
