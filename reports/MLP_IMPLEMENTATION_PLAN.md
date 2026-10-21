# Kế hoạch cài đặt MLP — chưa chạy mô hình

Ngày lập: 21/10/2026. Nhánh làm việc: `feat/mlp`.

Trạng thái: **mới lập kế hoạch; chưa cài dependencies, chưa chạy mô hình**.

## 1. Mục tiêu và điểm dừng

Chuẩn bị môi trường riêng trong repo để dùng MLP đã có, giữ đúng mô tả trong [báo cáo chuyển từ DOCX](../Bao_cao_y_tuong_Deep_Learning_DDI_hoan_chinh.md). Hoàn thành khi dependencies được cài, kiểm tra dependency và cú pháp đạt, hướng dẫn dùng MLP rõ ràng, và chưa thực hiện bất kỳ lượt chạy model nào.

Trong giai đoạn cài đặt này:

- Được tạo virtual environment, cài dependencies, kiểm tra metadata package và cú pháp Python.
- Không khởi tạo model, gọi forward pass, compile model, train, evaluate, predict, smoke hoặc chạy notebook.
- Không preprocessing dữ liệu, tạo split, checkpoint, metrics hoặc thư mục run.
- Không viết lại MLP, đổi kiến trúc, thêm framework, giao diện, script train riêng hay cấu hình mới.

## 2. Hiện trạng đã đối chiếu

| Thành phần | Thực tế trong checkout | Việc cần làm |
| --- | --- | --- |
| MLP | `model.py:create_model('mlp')` đã có mạng đầy đủ | Dùng lại, không triển khai mạng thứ hai |
| Dữ liệu theo batch | `model.py:dataset()` giữ đầu vào MLP hai chiều và cast sang float32 theo batch | Giữ nguyên |
| Huấn luyện | `model.py:train()` có optimizer, class weights, callbacks và lưu artifacts | Chưa gọi |
| CLI | Có `--architecture mlp`; mặc định vẫn là CNN–BiLSTM | Luôn ghi rõ cờ MLP và output riêng khi chạy sau này |
| Dự đoán | `predict.py:predict()` đọc kiến trúc từ config của run và hỗ trợ MLP | Giữ nguyên, chưa gọi |
| Kiểm thử | Có test save/load và forward pass cho cả hai kiến trúc | Giữ lại, hoãn thực thi |
| Dependencies | Có `requirements.txt` và `requirements-dev.txt` | Cài từ các file hiện có |
| Môi trường hiện tại | Python 3.12.3; chưa có `.venv`; chưa tìm thấy dependencies dự án | Tạo `.venv` riêng |
| Dữ liệu mới và checkpoint | Chưa có `data/processed/` hoặc `runs/` | Không cần tạo để nghiệm thu cài đặt |
| Validation đã ghi | `VALIDATION.md` mô tả lần chạy macOS ngày 19/10/2026 | Không coi là kết quả của môi trường hiện tại |

Kết luận: phần còn thiếu trước mắt là **cài môi trường và xác nhận việc cài**, không phải mã kiến trúc MLP. Chưa có bằng chứng cần sửa logic model cho yêu cầu này.

## 3. Đặc tả MLP giữ nguyên

```text
Input (4096,)
→ Dense 512, ReLU
→ Dropout 0.3
→ Dense 256, ReLU
→ Dropout 0.3
→ Dense 86, Softmax
```

Tên model hiện tại: `fingerprint_mlp`. Không thêm L2 vào MLP chỉ vì phần mô tả huấn luyện chung nhắc L2; regularizer hiện nằm ở CNN–BiLSTM.

Hợp đồng dữ liệu cần giữ khi chạy ở giai đoạn sau:

- Hai thuốc có thứ tự A rồi B; dùng chung `preprocess.py:pair_features()` cho train và predict.
- Mỗi thuốc dùng RDKit fingerprint 2048 bit, `maxPath=5`; nối thành 4096 bit.
- NPZ mới lưu `X=(N,4096)` dạng uint8, các giá trị 0 hoặc 1; pipeline train cast từng batch sang float32. Không reshape MLP thành `(N,4096,1)`.
- Nhãn CSV nguồn là số nguyên 1–86; `y=(N,)` trong NPZ là số nguyên 0–85; dự đoán trả nhãn nguồn 1–86.
- Bộ train phải có đủ 86 lớp. Không one-hot nhãn vì loss hiện là sparse categorical cross entropy.
- Khi chạy dữ liệu sau này, dùng split mới từ `preprocess.py`, có `metadata.json`; không dùng NPZ cũ ở gốc vì đã ghi nhận trùng cặp giữa các split.

Cấu hình train đã có để dùng về sau: Adam, learning rate `0.000173`, clipnorm `1.0`, batch size `32`, tối đa `50` epoch, seed `42`. Class weights chỉ tính từ train; checkpoint/early stopping/LR scheduler theo validation loss. Đây là cấu hình sẵn có, chưa chạy hoặc tuning trong giai đoạn cài.

## 4. Phạm vi file và callers

| File / thành phần | Xử lý trong giai đoạn cài đặt |
| --- | --- |
| `model.py` | Không sửa nếu không phát hiện lỗi cài đặt cụ thể. Giữ `create_model`, `load_data`, `dataset`, `train` và CLI hiện có |
| `predict.py` | Không sửa; tiếp tục đọc `architecture` từ config của checkpoint |
| `preprocess.py`, `check_data.py` | Không sửa hoặc chạy; là phần chuẩn bị/kiểm tra dữ liệu của giai đoạn sau |
| `tests/test_pipeline.py` | Không thêm test trùng chức năng; không chạy trong giai đoạn này |
| `requirements.txt`, `requirements-dev.txt` | Dùng nguyên các ràng buộc hiện có. Chỉ sửa nếu resolver chứng minh có xung đột thật, kèm nguyên nhân cụ thể |
| `requirements-lock-macos-py39.txt` | Giữ nguyên; không dùng làm lock cho Linux/Python 3.12 |
| `.gitignore` | Đã loại `.venv/`, `data/processed/`, `runs/`; không cần bổ sung |
| `README.md` | Khi thực thi cài đặt, thêm một ghi chú ngắn phân biệt bước cài với bước chạy và nêu interpreter cho IDE |
| `progress.md` | Khi cài thành công, ghi môi trường đã cài nhưng model chưa chạy trên môi trường này |
| `VALIDATION.md` | Khi cài thành công, nối thêm mục ngày thực tế với Python/package versions và check đã chạy; giữ nguyên kết quả macOS lịch sử |
| CSV/NPZ gốc, notebook, báo cáo gốc | Không sửa, di chuyển hoặc xóa |

Không cần module MLP riêng, file cấu hình hyperparameter, entrypoint mới hoặc đổi mặc định CNN–BiLSTM. Tái dùng các entrypoint hiện tại giữ cả hai kiến trúc tương thích.

## 5. Trình tự thực hiện cài đặt

Các lệnh dưới đây là **hướng dẫn cho lượt thực thi tiếp theo**, chưa được chạy khi lập kế hoạch. Chạy từ thư mục gốc repo trên shell Bash.

### Bước 1 — xác nhận checkout và interpreter

```bash
git branch --show-current
git status --short
python3 --version
```

Điều kiện: đúng nhánh `feat/mlp`; ghi nhận các thay đổi sẵn có của người dùng và không ghi đè chúng. README hiện chỉ định Python 3.9–3.12; dùng Python 3.12.3 đang có để tránh cài thêm interpreter. Nếu interpreter thực tế đã đổi ra ngoài khoảng này, chọn interpreter trong khoảng của repo trước khi tạo venv.

### Bước 2 — tạo môi trường và cài dependencies

Checkout lúc lập kế hoạch chưa có `.venv`. Nếu khi thực thi đã có thư mục này, kiểm tra nguồn và interpreter trước, không tạo đè hoặc xóa tự động.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` đã bao gồm `requirements.txt`, nên chỉ cần một lượt cài. Bộ hiện tại gồm NumPy, pandas, scikit-learn, matplotlib, RDKit, TensorFlow 2.16.x, Keras 3.10.x và pytest. Cài pytest để sẵn cho giai đoạn sau, chưa chạy pytest.

Không cài vào Python hệ thống, không dùng `sudo pip`, không thêm bộ CUDA/GPU trong phạm vi này. Không tự nâng TensorFlow hoặc nới các ràng buộc phiên bản để làm lệnh cài hết lỗi.

### Bước 3 — kiểm tra việc cài mà không thực thi model

```bash
.venv/bin/python -m pip check
.venv/bin/python -m pip list --format=freeze
.venv/bin/python - <<'PY'
from pathlib import Path

paths = [
    Path('model.py'), Path('preprocess.py'), Path('predict.py'),
    Path('check_data.py'), Path('tests/test_pipeline.py'),
]
for path in paths:
    compile(path.read_text(encoding='utf-8'), str(path), 'exec')
print('Syntax OK; project modules were not imported or executed.')
PY
```

`compile()` ở đây chỉ kiểm tra cú pháp nguồn, không phải `model.compile()`, không tạo model hoặc `.pyc`. `pip check` kiểm tra dependency metadata; nó không chứng minh TensorFlow/RDKit sẽ import và chạy đúng trên máy này. Ghi rõ giới hạn đó trong kết quả nghiệm thu.

### Bước 4 — chuẩn bị IDE và ghi nhận kết quả

1. Chọn interpreter `<repo>/.venv/bin/python` trong IDE; không cần commit cấu hình riêng của IDE.
2. Cập nhật các ghi chú README/progress/VALIDATION theo bảng phạm vi file. Chỉ ghi kết quả thực tế của lượt cài, không sao chép “8 passed” thành kết quả mới.
3. Kiểm tra `git diff --check`, `git diff --stat` và `git status --short`; bảo đảm chỉ có thay đổi tài liệu hoặc sửa dependency có lý do cụ thể.
4. Dừng sau khi báo kết quả cài đặt. Không chạy model để “kiểm tra nhanh”.

## 6. Các lệnh không chạy trong giai đoạn cài đặt

| Lệnh / thao tác | Lý do hoãn |
| --- | --- |
| `python preprocess.py` | Xử lý dữ liệu thật và ghi split; nằm ngoài phần cài môi trường |
| `python check_data.py` | Đọc/audit split; chưa có split mới và chưa đến giai đoạn chạy dữ liệu |
| `python model.py ...` | CLI luôn gọi `train()` sau khi parse args; không có chế độ chỉ cài |
| `python model.py --smoke ...` | Vẫn train 1 epoch rồi evaluate/predict test |
| `python predict.py ...` | Cần checkpoint và thực thi inference |
| `python -m pytest ...` | Suite có khởi tạo model, forward pass và save/load; test dữ liệu cũng tạo split |
| Import và gọi `create_model('mlp')` | Là khởi tạo model, chưa cần cho nghiệm thu installation-only |
| Chạy notebook cũ | Không thuộc pipeline chính thức và có bước gọi dịch vụ ngoài |

Không tạo sẵn `runs/mlp-full` hoặc `runs/smoke-mlp`: `train()` yêu cầu thư mục output chưa tồn tại (`exist_ok=False`).

## 7. Tiêu chí nghiệm thu

- [ ] Đang ở nhánh `feat/mlp`; không làm mất thay đổi hoặc dữ liệu có sẵn.
- [ ] Có `.venv` riêng với interpreter đúng khoảng repo quy định; IDE có thể chọn interpreter này.
- [ ] `pip install -r requirements-dev.txt` thành công, `pip check` không báo dependency thiếu/xung đột.
- [ ] Ghi lại phiên bản Python và package thực tế; kiểm tra cú pháp các file nêu trên đạt.
- [ ] Dùng nguyên kiến trúc MLP và các callers hiện có; nếu không có lỗi thì không sửa code.
- [ ] Tài liệu phân biệt rõ “đã cài môi trường” với “đã kiểm thử model/đã train”.
- [ ] Không sinh dữ liệu processed, run, checkpoint hoặc metrics; không thực thi model hay test runtime.
- [ ] Diff hợp lệ và không chứa `.venv`, dữ liệu sinh hoặc thay đổi ngoài phạm vi.

Đạt các mục này có nghĩa **đã cài và chuẩn bị xong**, chưa thể kết luận model hoạt động trên môi trường mới hoặc có chất lượng dự đoán tốt.

## 8. Khi có lỗi và cách hoàn tác

- Nếu không tạo được venv: ghi lỗi cụ thể về interpreter/venv; xử lý phần môi trường đó trước, không cài packages vào Python hệ thống để thay thế.
- Nếu cài package thất bại do mạng, dung lượng, thiếu wheel hoặc xung đột: ghi package, interpreter/platform và thông báo lỗi; xử lý nguyên nhân đó trước khi cân nhắc đổi requirements. Không bỏ qua dependency hoặc tuyên bố cài xong khi pip còn lỗi.
- Nếu đã phải đổi requirements để sửa lỗi thật: mô tả ràng buộc đã đổi, tác động lên cả MLP/CNN–BiLSTM/predict và kiểm tra dependency lại. Kiểm thử runtime vẫn hoãn theo yêu cầu.
- Nếu hủy việc cài: chỉ loại bỏ `.venv` được tạo mới trong chính lượt cài và hoàn tác các chỉnh sửa do lượt đó tạo. Giữ nguyên venv có sẵn, dữ liệu và thay đổi của người dùng; không dùng `git reset --hard` hoặc `git clean`.

## 9. Giai đoạn chạy sau này — chưa thực hiện

Khi chuyển sang yêu cầu chạy, thứ tự dự kiến là: kiểm thử hiện có → preprocessing → audit split → smoke MLP → train đầy đủ → đánh giá/dự đoán. Chọn thư mục output mới cho từng run.

Có hai giới hạn phải xử lý đúng khi thiết kế thực nghiệm sau này:

1. `train()` hiện tự reload checkpoint, evaluate và predict **test ngay sau fit**, kể cả smoke. Không dùng nó như một lệnh chỉ train/validation. Nếu cần chọn giữa nhiều kiến trúc/cấu hình trước khi mở test, cần điều chỉnh luồng đánh giá ở task chạy thực nghiệm đó.
2. Macro F1 hiện được xuất cho test; training history chưa có validation Macro F1. So sánh kiến trúc theo validation Macro F1 cần bước đánh giá validation riêng ở giai đoạn thực nghiệm, không suy ra từ accuracy hoặc test Macro F1.

Thiếu `data/processed/`, checkpoint, nguồn dữ liệu đã xác minh và mô tả 86 nhãn không chặn việc cài môi trường. Chúng vẫn là công việc còn lại trước khi chạy và diễn giải kết quả nghiên cứu; không đưa việc giải quyết chúng vào lượt cài đặt này.
