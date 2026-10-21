# MLP: quyết định triển khai và quy trình Colab

Ngày đối chiếu code: 21/10/2026. Tài liệu này ghi kiến thức cần dùng cho repo; chưa có kết quả chạy mới trên Colab.

## Tham khảo được gì từ CNN–BiLSTM?

| Nội dung | CNN–BiLSTM | MLP và cách tái dùng |
| --- | --- | --- |
| Đầu vào | Fingerprint `(batch,4096,1)` | Cùng fingerprint, giữ `(batch,4096)` |
| Kiến trúc | 4 khối Conv1D, 2 BiLSTM, Dense 256, Softmax | Dense 512/256, Dropout 0.3 sau mỗi Dense, Softmax 86 |
| Regularization | Dropout và L2 ở Dense 256 | Giữ Dropout hiện có; không thêm L2 chỉ để giống CNN |
| Huấn luyện | Adam, class weights, clipnorm 1.0, callbacks | Dùng chung `train()`; không viết script train MLP riêng |
| Checkpoint | Tốt nhất theo `val_loss` | Dùng cùng quy tắc; so sánh validation Macro F1 của các checkpoint |
| Dự đoán | Thêm trục kênh theo config | Dùng chung `predict.py`, không reshape MLP thành chuỗi |

Nguồn trong repo: [create_model/dataset/train](../model.py), [dự đoán](../predict.py), [xử lý dữ liệu](../preprocess.py), [tests](../tests/test_pipeline.py). Notebook `DataPreprocessing.ipynb` là tham khảo cũ; không dùng lại các cell ghi đè CSV ở gốc.

## Các hợp đồng và kiến thức cần giữ

- Canonical hóa SMILES trước khi deduplicate và split; SMILES sai phải báo lỗi. Cặp A,B giữ thứ tự, không tự gộp với B,A. RDKit cung cấp canonical SMILES và fingerprint; repo dùng RDKit fingerprint `maxPath=5`, 2048 bit mỗi thuốc rồi nối thành 4096 bit. Xem [tài liệu RDKit](https://www.rdkit.org/docs/GettingStartedInPython.html).
- NPZ mới lưu fingerprint nhị phân `uint8` để giảm RAM; `dataset()` chuyển sang float32 theo batch. Không nạp toàn bộ thành float32 hoặc one-hot nhãn để làm giống một ví dụ bên ngoài.
- Nhãn CSV là 1–86; preprocessing trừ 1 thành 0–85; predict cộng 1 để trả mã nguồn. Softmax 86 đầu ra đi với sparse categorical cross entropy, không chuyển thành sigmoid/multilabel khi chưa đổi bài toán.
- Cặp nhiều nhãn được cách ly vào `conflicting_pairs.csv`, không chọn ngẫu nhiên một nhãn. Split stratified 64/16/20 với seed 42 sau khi làm sạch; cả hai kiến trúc dùng cùng split và mapping.
- Class weights chỉ tính từ train. Validation/test giữ phân bố tự nhiên. Lớp hiếm có ít mẫu nên chỉ số từng lớp dao động mạnh; không kết luận chỉ từ accuracy.
- Macro F1 trung bình đều F1 các lớp, weighted F1 cân theo số mẫu; micro F1 bằng accuracy ở bài toán multiclass single-label này. Repo cố định đủ 86 lớp khi tính báo cáo, kể cả smoke. Xem [scikit-learn F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).
- `val_loss` chọn checkpoint trong một run; `validation_metrics.json` đo checkpoint đó để chọn giữa các run. Không đánh đồng hai tiêu chí và không coi file này là lịch sử validation F1 từng epoch. Xem [ModelCheckpoint](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint).
- Fingerprint là vector bit băm, không phải thứ tự nguyên tử hay chuỗi thời gian. Lợi ích của CNN–BiLSTM trên nó là giả thuyết cần đo so với MLP; chưa có kết luận kiến trúc nào tốt hơn.
- Không trùng cặp cùng thứ tự giữa split vẫn cho phép cùng một thuốc và cặp đảo chiều xuất hiện ở tập khác. Đây là đánh giá warm-start trên cặp mới, chưa phải cold-start trên thuốc mới.

## Chạy trên Colab ở giai đoạn sau

**Chỉ thực hiện các bước dưới đây trên Colab sau khi phần code đã hoàn thiện và chuyển đúng bản code mới lên đó. Không train local; không bấm Run all trước khi kiểm tra môi trường và tests.**

### 1. Chuẩn bị phiên và code

Chọn hosted runtime GPU trong Colab, đưa checkout hiện tại vào `/content/prj-DL`, gồm script, tests, requirements và CSV nguồn. Thay đổi chưa commit/push ở local không tự xuất hiện khi clone GitHub; cần chuyển đúng bản code. Không cần chuyển `.venv`, NPZ legacy hay chạy notebook cũ.

```python
%cd /content/prj-DL
import sys
print(sys.version)
assert (3, 9) <= sys.version_info[:2] <= (3, 12), 'Cần Python 3.9–3.12 cho dependencies hiện tại'
```

Colab thay đổi runtime và thư viện có sẵn theo thời gian; có tùy chọn runtime phiên bản cũ khi cần. Kiểm tra phiên thực tế thay vì giả định TensorFlow có sẵn phù hợp với repo. Xem [Colab runtime versions](https://research.google.com/colaboratory/runtime-version-faq.html).

### 2. Cài và xác nhận môi trường Colab

```python
%pip install -r requirements-dev.txt "tensorflow[and-cuda]>=2.16,<2.17"
%pip check
```

Giữ nguyên ràng buộc NumPy, Keras và TensorFlow của repo; extra `and-cuda` bổ sung thư viện GPU theo [hướng dẫn TensorFlow](https://www.tensorflow.org/install/pip). Không dùng lock macOS cho Linux. Nếu đổi package đã import, restart session rồi quay lại thư mục repo trước khi kiểm tra tiếp. Nếu resolver/pip check lỗi, xử lý môi trường trước khi train; chưa xác nhận bộ cài này trên phiên Colab thực tế.

```python
import tensorflow as tf
print('TensorFlow:', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
assert gpus, 'Chưa có GPU cho TensorFlow; dừng và xử lý runtime Colab'
```

Lưu Python, `pip freeze`, thông tin GPU và phiên bản code cùng kết quả. GPU được cấp tùy phiên; không cố định tên GPU hay thời gian chạy trước khi kết nối.

### 3. Xác nhận code, dữ liệu và smoke

Các lệnh `!` sau là cell shell trong Colab, không phải lệnh chạy local:

```python
!python -m pytest -q
```

Chỉ tiếp tục khi tests đạt; không chạy các cell sau nếu pytest lỗi.

```python
!python preprocess.py
!python check_data.py
```

Chỉ tiếp tục nếu preprocessing/audit thành công, cả ba split đủ lớp và không trùng cặp chuẩn hóa cùng thứ tự. Dùng chung `data/processed/` cho hai mạng, không chia lại giữa các run.

```python
!python model.py --smoke --architecture mlp --skip-test --output runs/smoke-mlp
!python model.py --smoke --architecture cnn-bilstm --skip-test --output runs/smoke-cnn
```

Smoke chạy 1 epoch trên mẫu nhỏ, chỉ nghiệm thu pipeline. Không dùng smoke Macro F1 chọn mô hình hay báo cáo chất lượng; không mở test ở bước này. Mỗi run dùng output mới, không tạo thư mục đó trước.

### 4. Train và chọn mô hình bằng validation

Chỉ chạy sau khi code/tests/dữ liệu/smoke đạt trên Colab:

```python
!python model.py --architecture mlp --skip-test --output runs/mlp-full --epochs 50 --batch-size 32 --seed 42
!python model.py --architecture cnn-bilstm --skip-test --output runs/cnn-full --epochs 50 --batch-size 32 --seed 42
```

Đọc `validation_metrics.json` từng run, chọn theo Macro F1 đã định trước; ghi loss, accuracy, weighted F1, thời gian và tài nguyên kèm theo. `val_loss` vẫn điều khiển checkpoint/early stopping. Nếu đổi cấu hình, dùng run mới và tiếp tục chọn bằng validation, chưa chấm test.

### 5. Chấm test cuối và giữ artifacts

Ví dụ **nếu MLP là run được chọn**:

```python
!python model.py --evaluate-only --output runs/mlp-full
```

Lệnh đọc checkpoint/config của run, không fit lại; xuất `metrics.json`, `confusion_matrix.npy`, `test_predictions.npz`. Không ghi đè artifacts test đã có. Giữ dữ liệu ở đúng `data_dir` ghi trong config khi chuyển hoặc mở lại phiên. Không dùng run smoke cho kết quả test cuối và không tiếp tục tuning theo test.

Lưu checkpoint/config/metadata/history/metrics ra Drive hoặc tải về; code chưa hỗ trợ resume train. Colab có thể xóa VM khi hết phiên, nên sao lưu checkpoint trong lúc chạy dài. Dùng dữ liệu trong `/content` để giảm I/O Drive; xem [Colab FAQ](https://research.google.com/colaboratory/faq.html).

Nguồn công bố dataset và diễn giải 86 nhãn còn thiếu; mã dự đoán và xác suất chưa phải hướng dẫn dùng thuốc.
