# DDI Deep Learning

Phân loại tương tác của **cặp thuốc có thứ tự** vào 86 lớp bằng TensorFlow/Keras.
Mô hình chính: 4 khối Conv1D → 2 BiLSTM → Dense → Softmax. Có MLP để đối chiếu trên cùng fingerprint và split.

## Cài đặt

Python 3.9–3.12, khuyến nghị Python 3.11. Tại thư mục project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

`requirements-lock-macos-py39.txt` ghi lại chính xác môi trường đã kiểm tra trên máy này; dùng `requirements.txt` khi cài trên nền tảng khác.

Windows dùng `.venv\Scripts\activate`. TensorFlow được giới hạn 2.16.x để thống nhất môi trường; GPU NVIDIA nên chạy Linux/Colab hoặc WSL2.

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
python model.py --smoke --architecture mlp --output runs/smoke-mlp
python model.py --smoke --output runs/smoke-cnn
```

Smoke dùng 1000/300/300 mẫu được lấy mẫu có seed, giữ ít nhất một mẫu mỗi lớp, và 1 epoch. **Các metrics smoke chỉ xác nhận pipeline chạy; không dùng báo cáo chất lượng mô hình.** Mỗi run phải dùng thư mục output mới để tránh ghi đè checkpoint.

## 3. Train đầy đủ

```bash
python model.py --output runs/cnn-full --epochs 50 --batch-size 32
python model.py --architecture mlp --output runs/mlp-full --epochs 50
```

CNN–BiLSTM trên 4096 bit có thể chậm trên CPU. Khuyến nghị Colab GPU cho full training. Fingerprint là vector bit băm, không có thứ tự chuỗi hóa học tự nhiên: CNN–BiLSTM là giả thuyết thực nghiệm cần so sánh với MLP, chưa thể khẳng định tốt hơn.

Có class weights tính chỉ từ train, gradient clipping, EarlyStopping, giảm learning rate theo validation loss và checkpoint tốt nhất. Dùng sparse categorical cross entropy, không cần one-hot. Seed cố định; thêm `--deterministic` nếu cần thực thi deterministic, có thể chậm hơn. Không tuning hyperparameter theo test.

Mỗi run xuất:

- `best_model.keras`: checkpoint theo validation loss.
- `config.json`, `data_metadata.json`: cấu hình và nguồn dữ liệu.
- `training.csv`, `history.json`, `training_history.png`: lịch sử train.
- `metrics.json`: loss, accuracy, micro/macro/weighted F1 và precision/recall/F1 từng lớp.
- `confusion_matrix.npy`: ma trận 86×86; chỉ số hàng/cột 0 ứng với nhãn nguồn 1.
- `test_predictions.npz`: nhãn đúng và dự đoán zero-based.

Ưu tiên đọc Macro F1 vì mất cân bằng lớp; micro F1 bằng accuracy trong bài toán single-label này. Chỉ đánh giá test sau khi chọn checkpoint bằng validation. Khi so sánh kiến trúc, nên chọn bằng validation trước và chỉ báo cáo test cuối cùng; không liên tục chọn model theo test.

## 4. Dự đoán

```bash
python predict.py --run runs/cnn-full --smiles1 'CCO' --smiles2 'CC(=O)O' --top-k 5
```

Trả nhãn nguồn 1–86 và xác suất softmax. Ví dụ SMILES trên chỉ minh họa cú pháp. Repository chưa cung cấp bảng mô tả ý nghĩa 86 nhãn, nên không tự diễn giải thành lời khuyên dùng thuốc. Xác suất chưa được hiệu chuẩn; đây là project nghiên cứu.

`DataPreprocessing.ipynb` là notebook cũ phục vụ tham khảo. Pipeline tái lập chính thức dùng các script ở trên, không cần chạy notebook hoặc gọi PubChem.

API tham khảo: [TensorFlow seed](https://www.tensorflow.org/api_docs/python/tf/keras/utils/set_random_seed), [RDKit fingerprints](https://www.rdkit.org/docs/GettingStartedInPython.html).
