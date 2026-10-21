# Handoff — 21/10/2026

## Giai đoạn hiện tại

Hoàn thiện code/docs trước; kiểm thử runtime và train chỉ trên Google Colab sau đó. Yêu cầu mới thay thế kế hoạch cài local. Chưa kết nối Colab, chưa chạy model trong lượt này.

## Trạng thái đã xác nhận

- Nhánh `feat/mlp`; code/docs được lưu bằng commit local theo yêu cầu, chưa push.
- MLP đã có trong `model.py`, dùng chung pipeline với CNN–BiLSTM; không cần triển khai mạng thứ hai.
- Đã bổ sung `validation_metrics.json` từ checkpoint tốt nhất, `--skip-test` khi lựa chọn và `--evaluate-only --output <run>` cho test cuối không fit lại. Lệnh train cũ vẫn tự chấm test nếu thiếu `--skip-test`.
- `evaluate-only` dùng config đã lưu và từ chối nếu bất kỳ artifact test nào đã có; tests đã bổ sung nhưng chưa chạy pytest.
- Đã đối chiếu code, kiểm tra cú pháp 5 file, 6 kiểm tra logic điều phối cô lập với mocks, skill validator và `git diff --check`; chưa xác nhận TensorFlow/RDKit/GPU runtime.
- `.venv` local được tạo trong lượt trước chỉ có pip, đã xóa theo yêu cầu; không cài tiếp dependencies local.
- Chưa có `data/processed/` hoặc `runs/`. CSV/NPZ tracked ở gốc là nguồn/legacy, không phải split mới.
- `VALIDATION.md` giữ nguyên bằng chứng macOS ngày 19/10/2026 và thêm mục kiểm tra tĩnh/logic ngày 21/10; không có kết quả train mới.

## Bước tiếp theo và giới hạn

1. Chuyển đúng phiên bản code mới lên Colab; clone GitHub chưa có các thay đổi chưa push ở local.
2. Trên Colab: kiểm tra Python/dependencies/GPU, chạy pytest; chỉ khi đạt mới preprocessing, audit và smoke với `--skip-test`.
3. Sau khi code và smoke đạt mới full training hai kiến trúc trên cùng split/seed. Chọn bằng validation Macro F1 rồi chấm test checkpoint đã chọn.
4. Chưa tự chạy các bước Colab trong lượt hoàn thiện code; không quay lại cài/train local. Chưa có cơ chế resume train.

## Tài liệu cần đọc theo nhu cầu

- [Kế hoạch](../reports/MLP_IMPLEMENTATION_PLAN.md): phạm vi code trước, Colab sau và checklist.
- [MLP_GUIDE.md](MLP_GUIDE.md): kiến thức tái dùng từ CNN–BiLSTM, hợp đồng dữ liệu, lệnh Colab tương lai.
- [RULES.md](RULES.md), [PROJECT_GUIDE.md](PROJECT_GUIDE.md): quy tắc và nguồn thông tin.
- [progress.md](../progress.md), [VALIDATION.md](../VALIDATION.md): tiến độ và bằng chứng theo ngày.
- [README.md](../README.md): CLI, artifacts và giới hạn sử dụng.

Handoff là snapshot hiện tại; progress là lịch sử; plan không phải bằng chứng đã thực thi.
