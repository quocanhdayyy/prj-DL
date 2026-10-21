# Kế hoạch hoàn thiện MLP và chạy trên Colab

Cập nhật: 21/10/2026. Nhánh: `feat/mlp`.

Yêu cầu hiện tại thay thế kế hoạch cài local trước đây: **hoàn thiện code và tài liệu trước; chỉ train trên Colab sau khi code sẵn sàng**. Chưa kết nối Colab, chưa chạy train trong lượt này.

## 1. Phạm vi hiện tại

- Tái dùng MLP và pipeline CNN–BiLSTM hiện có, không dựng mạng thứ hai hay đổi framework.
- Hoàn thiện đường chạy để chọn mô hình bằng validation trước khi chấm test.
- Viết test cho hành vi mới, kiểm tra tĩnh/logic điều phối tại local; xác nhận TensorFlow runtime trên Colab ở giai đoạn sau.
- Ghi kiến thức cần dùng và quy trình Colab tại [MLP_GUIDE.md](../docs/MLP_GUIDE.md).
- Dọn `.venv` local được tạo trong lượt cài trước; không cài dependencies nặng tại local.
- Giữ nguyên CSV/NPZ nguồn, notebook cũ và báo cáo chuyển từ DOCX. Không sinh split, checkpoint hay metrics trong lượt này.

## 2. Code và phần tham khảo từ CNN–BiLSTM

| Thành phần | Quyết định |
| --- | --- |
| `model.py:create_model('mlp')` | Giữ 4096 → Dense 512 ReLU → Dropout 0.3 → Dense 256 ReLU → Dropout 0.3 → Softmax 86 |
| `dataset()` | MLP dùng `(batch,4096)`; chỉ CNN thêm trục kênh; uint8 được cast float32 theo batch |
| `train()` dùng chung | Giữ Adam, class weights chỉ từ train, clipping, checkpoint, EarlyStopping và giảm learning rate theo `val_loss` |
| `preprocess.py`, `predict.py` | Dùng chung fingerprint/cặp có thứ tự; predict đọc kiến trúc từ config của run |
| Validation | Reload checkpoint tốt nhất rồi xuất `validation_metrics.json`, gồm Macro F1 để chọn giữa các run |
| Test | Dùng `--skip-test` khi smoke/train; `--evaluate-only --output <run>` chấm checkpoint đã chọn mà không fit lại |
| Tương thích | Lệnh train cũ không có `--skip-test` vẫn tự chấm test; hướng dẫn Colab luôn dùng cờ này ở bước lựa chọn |
| Tests | Giữ test save/load cả hai mạng; bổ sung kiểm tra không đọc test và đánh giá theo config, bảo vệ artifacts đã có |

Không sao chép Conv1D/BiLSTM hay L2 của CNN sang MLP: đó là thay đổi giả thuyết thực nghiệm, không phải điều kiện để dùng chung pipeline.

## 3. Thứ tự thực hiện và điểm dừng

| Giai đoạn | Nội dung | Trạng thái |
| --- | --- | --- |
| A — code và docs | Đối chiếu hai kiến trúc, hoàn thiện validation/test, viết tests và hướng dẫn | Code đã bổ sung; kiểm tra tĩnh/logic cô lập, chưa xác nhận TensorFlow runtime |
| B — chuẩn bị Colab | Chuyển đúng phiên bản code, kiểm tra Python/dependencies/GPU, chạy pytest | Chưa thực hiện |
| C — dữ liệu và smoke | Tạo split mới, audit; smoke hai kiến trúc với `--skip-test` | Chưa thực hiện; chỉ trên Colab sau khi B đạt |
| D — train và chọn | Train trên cùng split/seed; chọn run bằng validation Macro F1, ghi chi phí chạy | Chưa thực hiện; chỉ trên Colab sau khi code và smoke đạt |
| E — test cuối | Chấm checkpoint đã chọn; lưu kết quả và dự đoán | Chưa thực hiện; không tuning tiếp theo test |

Các lệnh B–E nằm trong [hướng dẫn Colab](../docs/MLP_GUIDE.md). Đó là hướng dẫn tương lai, không phải bằng chứng đã chạy hay lệnh tự chạy trong lượt này.

## 4. Tiêu chí hoàn thành phần code

- [x] Dùng lại MLP, callers và hợp đồng dữ liệu hiện có.
- [x] Có validation Macro F1 từ checkpoint đã chọn theo `val_loss`.
- [x] Có đường train/smoke bỏ qua test và đường chấm test riêng không fit lại.
- [x] Có tests cho hành vi mới và hướng dẫn lựa chọn mô hình trước test.
- [x] `.venv` local của lượt cài trước đã dọn; không đổi requirements.
- [x] Kiểm tra cú pháp, logic điều phối cô lập và diff tại local; không import project hay train.
- [ ] Pytest, TensorFlow/RDKit và GPU được xác nhận trên Colab.
- [ ] Smoke hai kiến trúc đạt trên Colab trước full training.

Phần triển khai đã chuẩn bị để chuyển lên Colab; chưa được gọi là runtime đã kiểm chứng. Kết quả macOS ngày 19/10 trong `VALIDATION.md` là lịch sử và không xác nhận thay đổi mới.

## 5. Giới hạn cần giữ khi thực nghiệm

- Dùng split mới có `metadata.json`; NPZ ở gốc là legacy có ghi nhận trùng cặp, không dùng để so sánh.
- Cặp có thứ tự A → B; source labels 1–86, labels lưu trong NPZ 0–85. Không one-hot, không tự gán mô tả lâm sàng cho mã lớp.
- Split hiện tại đánh giá cặp mới; thuốc có thể tái xuất hiện giữa các tập. Chưa chứng minh cold-start.
- Smoke chỉ kiểm tra pipeline. Macro F1 trên validation dùng chọn cấu hình; test dùng báo cáo cuối sau khi chốt.
- Chưa xác minh nguồn công bố dataset và ý nghĩa 86 nhãn. Phải bổ sung trước khi diễn giải kết quả nghiên cứu.
- Chưa có cơ chế resume train sau khi Colab mất phiên; checkpoint đã lưu không đồng nghĩa CLI hỗ trợ resume. Cần giữ artifacts ngoài ổ đĩa tạm của Colab.
