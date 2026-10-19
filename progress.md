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
