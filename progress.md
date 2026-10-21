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


## Trạng thái ngày 21/10/2026 — cài local đã dừng

- Theo yêu cầu ban đầu, đã tạo `.venv` bằng Python 3.12.3 và bắt đầu `pip install -r requirements-dev.txt`; lần thử trong sandbox bị chặn DNS. Lượt thử có quyền mạng sau đó tải được metadata và một số wheel, nhưng đã hủy theo phạm vi mới: môi trường đích để train là Google Colab, local không chạy runtime.
- Sau khi hủy, `.venv` chỉ có `pip==24.0`; dependency dự án chưa cài hoàn chỉnh. Không chạy `pip check`, compile, test, preprocessing hay model; không có kết quả local runtime. Giữ `.venv` hiện có, không xóa hoặc cài tiếp khi chưa có yêu cầu mới.
- Bước tiếp theo là chốt hướng dẫn/thực thi trên Colab theo scope người dùng; không suy diễn task local setup thành việc chạy remote.


## Trạng thái ngày 21/10/2026 — code trước, train Colab sau

- Theo yêu cầu mới, đổi plan từ cài local sang hoàn thiện code/docs; kiểm thử runtime, preprocessing, smoke và train chỉ thực hiện trên Colab ở giai đoạn sau.
- Tái dùng MLP và pipeline CNN–BiLSTM; bổ sung validation Macro F1 từ checkpoint tốt nhất, `--skip-test` và `--evaluate-only`. Chế độ đánh giá riêng dùng config đã lưu, không fit và không ghi đè artifacts test có sẵn. Lệnh train không có cờ mới vẫn giữ hành vi cũ.
- Thêm tests cho việc bỏ qua test, dùng đúng config khi chấm checkpoint và bảo vệ ba artifacts test. Chưa chạy pytest vì chưa có môi trường Colab.
- Thêm `docs/MLP_GUIDE.md` ghi phần tham khảo từ CNN–BiLSTM, fingerprint/shape/nhãn, mất cân bằng lớp, validation/test, giới hạn split và các bước Colab; đồng bộ README, plan, hướng dẫn repo và handoff.
- Đã xóa đúng `.venv` local tạo ở lượt trước (khoảng 16 MB, chỉ có pip); giữ nguyên dữ liệu, requirements, notebook và báo cáo gốc.
- Kiểm tra thực tế: compile nguồn 5 file đạt; 6 kiểm tra logic cô lập dùng mock đạt (không import project/TensorFlow, không train); skill validator và `git diff --check` đạt. Chi tiết giới hạn xem mục ngày 21/10 trong `VALIDATION.md`.
- Còn lại: chuyển đúng bản code lên Colab, kiểm tra dependencies/GPU và pytest trước khi smoke/full training; chưa có bằng chứng runtime mới hoặc kết quả chất lượng.
