# DDI Deep Learning

Phân loại tương tác của **cặp thuốc có thứ tự** vào 86 lớp bằng TensorFlow/Keras.
Mô hình chính: 4 khối Conv1D → 2 BiLSTM → Dense → Softmax. Có MLP để đối chiếu trên cùng fingerprint và split.

## Cài đặt

Quy trình hiện tại: **hoàn thiện code ở local, kiểm thử runtime và train trên Colab sau đó**. Không cần cài TensorFlow hoặc tạo `.venv` local cho bước chỉnh code. Xem [kế hoạch](reports/MLP_IMPLEMENTATION_PLAN.md) và [kiến thức triển khai/hướng dẫn Colab](docs/MLP_GUIDE.md).

Các lệnh pipeline dưới đây dành cho Colab đã chuẩn bị, không tự chạy trong giai đoạn hoàn thiện code. Dependencies hiện dùng Python 3.9–3.12, TensorFlow 2.16.x, Keras 3.10.x. `requirements-dev.txt` bao gồm dependencies chính và pytest. Lock macOS chỉ ghi môi trường lịch sử ngày 19/10/2026, không dùng cho Colab.

## 1. Chuẩn bị dữ liệu

```bash
python preprocess.py
```

Đọc `SMILES_DDI_filtered.csv` có `Smiles1,Smiles2,Label`; nhãn nguồn phải là số nguyên **1–86**.
Canonical hóa SMILES, loại dòng trùng, xuất các cặp có nhãn mâu thuẫn vào `data/processed/conflicting_pairs.csv` và cách ly chúng khỏi bài toán single-label. Không đảo thứ tự hai thuốc vì ý nghĩa tương tác có thể phụ thuộc thứ tự. SMILES sai làm dừng chương trình, không âm thầm thay bằng vector 0.

Chia stratified train/validation/test = **64%/16%/20%**, seed 42, sau khi làm sạch. Mỗi thuốc thành RDKit fingerprint 2048 bit (`maxPath=5`); ghép thành 4096 bit. NPZ dùng uint8 tiết kiệm bộ nhớ, nhãn 0–85; model chuyển từng batch sang float32. Metadata, CSV và NPZ mới nằm trong `data/processed/`; dữ liệu gốc được giữ nguyên.

**Không dùng NPZ cũ ở thư mục gốc:** đã phát hiện 56 cặp trùng train–validation, 66 cặp trùng train–test và 18 cặp trùng validation–test trước canonical hóa.

Đây là đánh giá trên cặp mới giữa các thuốc có thể đã xuất hiện ở train (warm-start), không chứng minh khả năng dự đoán thuốc hoàn toàn mới. Cặp đảo chiều có thể nằm khác split. Nếu cần đánh giá cold-start phải thiết kế split theo danh tính thuốc riêng.

## 2. Kiểm tra nhanh

```bash
python -m pytest -q
python model.py --smoke --architecture mlp --skip-test --output runs/smoke-mlp
python model.py --smoke --skip-test --output runs/smoke-cnn
```

Smoke với `--skip-test` dùng 1000 mẫu train/300 validation, được lấy mẫu có seed, giữ ít nhất một mẫu mỗi lớp, và 1 epoch. Nếu chấm test cho run smoke, chỉ dùng tối đa 300 mẫu test. **Các metrics smoke chỉ xác nhận pipeline chạy; không dùng báo cáo chất lượng mô hình.** Mỗi run phải dùng thư mục output mới để tránh ghi đè checkpoint.

## 3. Train đầy đủ

```bash
python model.py --skip-test --output runs/cnn-full --epochs 50 --batch-size 32
python model.py --architecture mlp --skip-test --output runs/mlp-full --epochs 50
```

Chỉ train sau khi code hoàn thiện và tests, audit dữ liệu, smoke đạt trên Colab GPU. Fingerprint là vector bit băm, không có thứ tự chuỗi hóa học tự nhiên: CNN–BiLSTM là giả thuyết thực nghiệm cần so sánh với MLP, chưa thể khẳng định tốt hơn.

Có class weights tính chỉ từ train, gradient clipping, EarlyStopping, giảm learning rate theo validation loss và checkpoint tốt nhất. Dùng sparse categorical cross entropy, không cần one-hot. Seed cố định; thêm `--deterministic` nếu cần thực thi deterministic, có thể chậm hơn. Không tuning hyperparameter theo test.

Mỗi run xuất:

- `best_model.keras`: checkpoint theo validation loss.
- `config.json`, `data_metadata.json`: cấu hình và nguồn dữ liệu.
- `training.csv`, `history.json`, `training_history.png`: lịch sử train.
- `validation_metrics.json`: loss, accuracy, micro/macro/weighted F1 và báo cáo từng lớp trên validation từ checkpoint tốt nhất.

Ưu tiên đọc Macro F1 vì mất cân bằng lớp; micro F1 bằng accuracy trong bài toán single-label này. Dùng `--skip-test` để không đọc/chấm test trong smoke và các run chọn cấu hình. Lệnh train cũ không có cờ này vẫn tự đánh giá test để giữ tương thích.

Sau khi chọn run bằng validation Macro F1, chấm test riêng, ví dụ nếu chọn MLP:

```bash
python model.py --evaluate-only --output runs/mlp-full
```

Lệnh dùng config/checkpoint đã lưu, không train lại; xuất `metrics.json`, `confusion_matrix.npy` (86×86; chỉ số 0 ứng với nhãn nguồn 1) và `test_predictions.npz` (nhãn zero-based). Từ chối ghi đè artifacts test có sẵn. Không tuning tiếp theo test. `--evaluate-only` trên run smoke vẫn chỉ chấm mẫu nhỏ, không dùng báo cáo chất lượng.

## 4. Dự đoán

```bash
python predict.py --run runs/cnn-full --smiles1 'CCO' --smiles2 'CC(=O)O' --top-k 5
```

Trả nhãn nguồn 1–86 và xác suất softmax. Ví dụ SMILES trên chỉ minh họa cú pháp. Repository chưa cung cấp bảng mô tả ý nghĩa 86 nhãn, nên không tự diễn giải thành lời khuyên dùng thuốc. Xác suất chưa được hiệu chuẩn; đây là project nghiên cứu.

`DataPreprocessing.ipynb` là notebook cũ phục vụ tham khảo. Pipeline tái lập chính thức dùng các script ở trên, không cần chạy notebook hoặc gọi PubChem.

API tham khảo: [TensorFlow seed](https://www.tensorflow.org/api_docs/python/tf/keras/utils/set_random_seed), [RDKit fingerprints](https://www.rdkit.org/docs/GettingStartedInPython.html).

## Hướng dẫn trong repo

Xem [AGENTS.md](AGENTS.md) và [chỉ mục tài liệu](docs/PROJECT_GUIDE.md) để biết quy tắc làm việc và nguồn thông tin chính của project.
# prj-DL
