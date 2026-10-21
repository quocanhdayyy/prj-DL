# Tiến độ

Đã thay pipeline cũ bằng các script có CLI và output riêng từng run:

- Chuẩn hóa cấu trúc phân tử, deduplicate trước stratified split.
- Cách ly cặp nhiều nhãn; giữ bài toán single-label 86 lớp.
- CNN–BiLSTM và MLP baseline; class weights, checkpoint, early stopping, LR scheduler.
- Sparse labels, fingerprint uint8 và cast theo batch.
- Xuất cấu hình, lịch sử, metrics từng lớp, confusion matrix và dự đoán test.
- Dự đoán từ hai SMILES với cùng fingerprint dùng khi train.
- Tests cho lỗi dữ liệu, split và lưu/đọc model.

Kết quả kiểm chứng thực tế xem `VALIDATION.md` sau khi chạy. Chưa có kết luận kiến trúc nào tốt hơn hoặc chất lượng full training. Không dùng kết quả smoke làm kết quả nghiên cứu.

## Trạng thái ngày 21/10/2026

- Đã có implementation MLP trong `model.py`; kế hoạch cài môi trường đã được ghi tại `reports/MLP_IMPLEMENTATION_PLAN.md`.
- Đã thêm hướng dẫn repo, quy tắc và skill cài đặt tại `AGENTS.md`, `docs/RULES.md`, `docs/PROJECT_GUIDE.md` và `.agents/skills/ddi-setup/SKILL.md`.
- Đã thêm `docs/HANDOFF.md` làm snapshot continuity; link nội bộ và `git diff --check` đạt trong lần cập nhật docs này.
- Chưa cài môi trường theo kế hoạch, chưa chạy model/runtime trong phase này. Checkout hiện chưa có `data/processed/` hoặc `runs/`.
- Bằng chứng của mục này: trạng thái Git `feat/mlp`, các file kế hoạch/hướng dẫn hiện có và kiểm tra thư mục artifact vắng mặt; không phải kết quả chạy model.
- Bước tiếp theo: chỉ khi user yêu cầu, thực hiện cài đặt theo plan; dừng trước preprocessing/test/model runtime nếu phạm vi vẫn installation-only.
- Điểm vào continuity: [docs/HANDOFF.md](docs/HANDOFF.md); quy tắc cập nhật docs/progress: [docs/RULES.md](docs/RULES.md).
