# Kiểm chứng ngày 19/10/2026

Môi trường thực tế: macOS ARM64, Python 3.9.6, TensorFlow 2.16.2, Keras 3.10.0, NumPy 1.26.4, RDKit 2025.9.2. Phiên bản đầy đủ trong `requirements-lock-macos-py39.txt`.

- `pip check`: không có dependency bị thiếu/xung đột.
- `python -m pytest -q`: **8 passed**; gồm SMILES sai, nhãn sai, canonical duplicates, nhãn mâu thuẫn, giữ thứ tự cặp, split không overlap, lưu/đọc cả hai model và lấy mẫu smoke khi có lớp singleton.
- `python preprocess.py`: chạy thành công trên dữ liệu thật. Cách ly 578 dòng thuộc 289 cặp nhiều nhãn; còn 191264 mẫu single-label.
- `python check_data.py`: cả ba split có 86 lớp; **0 canonical ordered-pair overlap** ở cả ba tổ hợp split.

| Split | Số mẫu | Lớp nhỏ nhất | Lớp lớn nhất |
|---|---:|---:|---:|
| Train | 122408 | 4 | 38987 |
| Validation | 30603 | 1 | 9747 |
| Test | 38253 | 1 | 12184 |

Hai smoke run trên dữ liệu thật, mỗi run 1000/300/300 mẫu, 1 epoch:

- `runs/smoke-mlp-v2`: train, validation, checkpoint reload, test evaluation, metrics và đồ thị hoàn tất.
- `runs/smoke-cnn`: CNN–BiLSTM train, validation, checkpoint reload, test evaluation, metrics và đồ thị hoàn tất.
- Lệnh dự đoán SMILES hoạt động với checkpoint smoke; output ghi rõ `smoke_model: true`.

Smoke chỉ kiểm tra pipeline. **Chưa train đầy đủ 50 epoch, chưa có kết quả chất lượng mô hình cuối cùng và chưa kết luận kiến trúc nào tốt hơn.** Lớp hiếm có một mẫu ở validation/test khiến metric từng lớp có độ bất định cao. Cần full training và so sánh bằng validation trước khi công bố kết quả test cuối.

Có warning từ thư viện về pyparsing deprecated và LibreSSL của Python hệ thống; các bước chạy local trên không bị lỗi. Khuyến nghị Python 3.11 cho môi trường mới.
