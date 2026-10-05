# Báo cáo bonus — Lab Ngày 18 (2D Perception)

Hai phần bonus đã làm: **4C** (val lật gương: `flip_idx` giải phẫu so với đồng nhất) và **Bài tập về nhà 3**
(export ONNX, đo latency trên CPU). Mọi số liệu dưới đây lấy từ notebook nộp kèm (`lab_2d_perception_student.ipynb`,
chạy toàn bộ trên SLURM, GPU RTX 3090, `ultralytics==8.4.171`, `torch 2.6.0+cu124`). Số liệu gốc nằm trong
`ket_qua.json` (`result_4c`) và `bonus_onnx_latency.json`.

---

## 1. Thí nghiệm 4C — metric nào đã che lỗi `flip_idx`?

**Thiết lập.** Hai model YOLO26n-pose, train giống hệt nhau trên tiger-pose (40 epoch, imgsz 640, batch 16, seed 0,
`fliplr=0.5` mặc định), chỉ khác `flip_idx`:

- **giải phẫu**: `[0, 1, 2, 3, 7, 6, 5, 4, 10, 11, 8, 9]` (đổi chỗ từng cặp `left_*` ↔ `right_*`);
- **đồng nhất**: `[0, 1, …, 11]` (YAML gốc).

Cả hai được chấm trên **val gốc** (53 ảnh, 100 % hổ quay phải, giống train 210/0) và **val lật gương** (cùng 53 ảnh lật
ngang, nhãn đổi theo quy ước giải phẫu, giả lập hổ quay trái lúc triển khai).

| Model | Val | Box mAP50-95 | Pose P | Pose R | Pose mAP50 | Pose mAP50-95 |
|---|---|---:|---:|---:|---:|---:|
| `flip_idx` giải phẫu | gốc | 0.905 | 0.999 | 1.000 | 0.995 | **0.424** |
| `flip_idx` giải phẫu | lật gương | 0.900 | 0.999 | 1.000 | 0.995 | **0.409** |
| `flip_idx` đồng nhất | gốc | 0.924 | 0.999 | 1.000 | 0.995 | **0.386** |
| `flip_idx` đồng nhất | lật gương | 0.908 | 0.772 | 0.774 | **0.657** | **0.265** |

**Đọc bảng.**

- Trên **val gốc**, hai model gần như không phân biệt được. Box mAP của model đồng nhất còn cao hơn (0.924 so với 0.905),
  Pose mAP50 cùng là 0.995 (bão hoà), P/R cùng ≈ 1. Chỉ Pose mAP50-95 chênh 0.038. Với một seed duy nhất, khoảng chênh này
  chưa đủ để kết luận là có lỗi.
- Trên **val lật gương**, model giải phẫu gần như giữ nguyên (mAP50-95 0.424 → 0.409, −3.5 %; mAP50 vẫn 0.995). Model đồng
  nhất sụp: Pose mAP50 0.995 → 0.657, mAP50-95 0.386 → 0.265 (−31 %), P/R rơi xuống ~0.77. Nguyên nhân là nó đã học
  "chân phía camera = `right_*`" thay vì chân phải giải phẫu, nên khi hổ quay trái thì toàn bộ chân trái/phải bị đổi chỗ.

**Metric nào đã che lỗi.**

1. **Box mAP**: box không có khái niệm trái/phải, nên box mAP không thể nhìn thấy lỗi `flip_idx`.
2. **Pose mAP50 trên val gốc**: bão hoà ở 0.995 cho cả hai model. Ngưỡng OKS 0.5 rộng tới mức hai cặp chân (vốn nằm gần
   nhau trên ảnh hổ nhìn ngang) đổi chỗ cho nhau vẫn được tính là đúng.
3. Gốc rễ không phải ở metric mà ở **tập val**: val có cùng phân phối hướng quay với train (100 % quay phải). Trên tập
   này, quy ước sai của model đồng nhất *trùng* với nhãn, nên không metric nào đo được lỗi. Lỗi chỉ lộ ra khi phân phối
   lúc đánh giá khác lúc train (val lật gương).

**Thiết kế tập val tốt hơn.**

- Có cả hai hướng quay. Nếu thiếu dữ liệu thì thêm val lật gương với nhãn giải phẫu như trên, và báo cáo metric **theo
  từng lát** (quay trái / quay phải) chứ không chỉ một con số trung bình.
- Theo dõi metric nhạy với việc đổi chỗ: tỉ lệ ảnh có OKS tăng khi đổi `left_*` ↔ `right_*`. Ô phân tích lỗi 4B đếm được
  17/53 ảnh như vậy ngay cả với model giải phẫu. Theo dõi thêm sai số theo từng keypoint và mAP50-95 thay cho mAP50.
- Chia train/val theo **video/cảnh** chứ không theo khung hình. Toàn bộ ảnh tiger-pose là khung hình liền nhau của cùng
  một con hổ, cùng một nền, nên val gần như trùng train và mAP50 bão hoà.
- Dùng `kpt_oks_sigmas` riêng cho từng keypoint (khoá này có trong YAML dataset của Ultralytics 8.4) để OKS phản ánh đúng
  độ mơ hồ của từng điểm (xem Q12).

---

## 2. Bài tập về nhà 3 — export ONNX, latency trên CPU

**Thiết lập.** `yolo26n.pt` được export sang ONNX (opset 18, input tĩnh 1×3×640×640) theo hai cách:

| Graph | Lệnh export | Output | Hậu xử lý |
|---|---|---|---|
| one-to-many | `export(format="onnx", nms=None)` | `(1, 84, 8400)` | `non_max_suppression` của Ultralytics, IoU 0.7 (giống 1C) |
| one-to-one | `export(format="onnx", nms=False)` | `(1, 300, 6)`, top-k nằm trong graph | chỉ lọc theo conf |

Chạy bằng ONNX Runtime 1.23.2, `CPUExecutionProvider`, CPU Intel Core i9-9920X, 12 luồng (đúng số core SLURM cấp; dùng
chung cho ORT và torch). Đo riêng ba giai đoạn: tiền xử lý (letterbox 640×640), ONNX inference, hậu xử lý. Có 10 vòng
warm-up chung cho mọi cấu hình, sau đó lấy **trung vị của 50 lần** cho mỗi cấu hình. Có hai ảnh: `bus.jpg` và một
**cảnh đông** ghép 3×3 ảnh `bus.jpg` (khoảng 36 người + 9 xe buýt, được thu nhỏ về 640×640).

| Ảnh | Cấu hình | preprocess (ms) | ONNX (ms) | postprocess (ms) | tổng (ms) | ứng viên qua conf | số box |
|---|---|---:|---:|---:|---:|---:|---:|
| bus.jpg | one-to-many + NMS, conf 0.25 | 2.88 | 33.90 | 15.97 | 52.76 | 48 | 5 |
| bus.jpg | one-to-many + NMS, conf 0.001 | 2.86 | 34.57 | 23.45 | 60.89 | 620 | 186 |
| bus.jpg | one-to-one NMS-free, conf 0.25 | 2.93 | 24.76 | 0.26 | 27.96 | 5 | 5 |
| bus.jpg | one-to-one NMS-free, conf 0.001 | 2.91 | 24.19 | 0.28 | 27.38 | 177 | 177 |
| cảnh đông 3×3 | one-to-many + NMS, conf 0.25 | 6.23 | 33.32 | 17.76 | 57.31 | 315 | 41 |
| cảnh đông 3×3 | one-to-many + NMS, conf 0.001 | 6.18 | 28.35 | 30.33 | 64.86 | 2199 | 300 |
| cảnh đông 3×3 | one-to-one NMS-free, conf 0.25 | 3.28 | 23.40 | 0.26 | 26.95 | 36 | 36 |
| cảnh đông 3×3 | one-to-one NMS-free, conf 0.001 | 3.27 | 23.35 | 0.28 | 26.90 | 300 | 300 |

**Nhận xét.**

1. **Trên CPU, NMS là một phần đáng kể của latency.** Hậu xử lý của one-to-many + NMS mất 16–30 ms, tức 30–47 % tổng thời
   gian. Head one-to-one chỉ mất 0.26–0.28 ms, nhanh hơn khoảng 60–110 lần. Cả pipeline one-to-one mất khoảng 27–28 ms so với
   53–65 ms của one-to-many, nhanh hơn khoảng 1.9–2.4 lần. Trên GPU (bảng 1C) hai head chỉ chênh khoảng 0.5 ms trong tổng
   khoảng 10 ms. Lợi ích của NMS-free vì thế lộ rõ nhất trên CPU/edge, đúng như bài giảng.
2. **Chi phí NMS tăng theo số ứng viên.** Với `bus.jpg`, hậu xử lý tăng từ 16.0 ms (48 ứng viên, conf 0.25) lên 23.5 ms (620
   ứng viên, conf 0.001). Với cảnh đông, nó tăng từ 17.8 ms (315 ứng viên) lên 30.3 ms (2199 ứng viên). Ngay cả khi chỉ có 48
   ứng viên vẫn tốn khoảng 16 ms, nên một phần lớn chi phí nằm ở khâu xử lý trước NMS trên CPU: chuyển vị tensor 84×8400,
   lấy max theo 80 class, lọc conf. Head one-to-one có latency gần như **không đổi** bất kể conf hay độ đông, vì top-300 đã
   nằm sẵn trong graph. Điều này quan trọng khi cần latency ổn định (tính mAP ở conf 0.001, cảnh đông người).
3. **Số box không giống hệt nhau** ở cảnh đông, conf 0.25: one-to-many + NMS giữ 41 box, one-to-one giữ 36 box. Mình chưa
   kiểm tra trường hợp nào đúng hơn trên ảnh ghép; con số này không dùng để đánh giá độ chính xác. Ở conf 0.001, cả hai
   đều chạm trần `max_det = 300`.
4. **Lưu ý khi đọc số.** Cột ONNX của graph one-to-many luôn chậm hơn graph one-to-one khoảng 5–11 ms, dù hai graph có
   cùng backbone. Cột preprocess dùng cùng một hàm cho cả hai graph nhưng lúc ra khoảng 3 ms, lúc khoảng 6 ms, và thay đổi
   giữa các lần chạy: ở lần chạy trước, `bus.jpg` với one-to-many mất khoảng 5.9 ms, lần này chỉ 2.9 ms. Mình **chưa kiểm
   chứng** nguyên nhân. Giả thuyết là các luồng OpenMP của torch (dùng trong NMS) tiếp tục chiếm CPU một lúc sau khi xong,
   tranh core với ONNX Runtime và bước tiền xử lý ở lần chạy kế tiếp, cộng thêm tải khác trên node. Vì vậy so sánh đáng tin
   nhất là cột postprocess, và xu hướng của cột này giữ nguyên qua ba lần chạy. Số tuyệt đối còn phụ thuộc CPU và tải của
   node: lần chạy thử trên một máy i7-7820X dùng chung cho cùng xu hướng nhưng số khác.

**Kết luận cho camera cổng.** Nếu chạy trên CPU/edge, head one-to-one cho pipeline nhanh hơn khoảng 2 lần, latency gần như
không đổi theo số người trong khung hình, và graph ONNX tự chứa toàn bộ, không cần cài NMS ở phía triển khai.
