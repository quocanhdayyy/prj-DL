# Báo cáo ý tưởng đề tài môn Deep Learning

## Phân loại tương tác giữa hai thuốc từ cấu trúc phân tử bằng CNN và BiLSTM

**Sinh viên thực hiện:** ........................................................  
**Mã sinh viên và lớp:** ........................................................  
**Giảng viên hướng dẫn:** .....................................................  
**Ngày báo cáo:** 19/10/2026

### Tóm tắt đề xuất

Em đề xuất xây dựng mô hình học sâu để phân loại loại tương tác của một cặp thuốc dựa trên biểu diễn cấu trúc phân tử của hai thuốc. Đầu vào là hai chuỗi SMILES; mỗi chuỗi được chuyển thành fingerprint RDKit 2.048 bit, sau đó ghép thành vector 4.096 chiều. Mô hình chính kết hợp CNN một chiều và BiLSTM, đầu ra là phân bố xác suất trên 86 lớp tương tác đã có trong dữ liệu. Em sẽ so sánh mô hình này với mạng MLP trên cùng bộ dữ liệu để đánh giá giá trị của việc kết hợp CNN và BiLSTM.

Mục tiêu của đề tài là xây dựng quy trình dữ liệu và thực nghiệm có thể tái lập, phân tích kết quả trên dữ liệu mất cân bằng và chỉ ra giới hạn của mô hình. Hiện quy trình đã chạy thử thành công; chất lượng dự đoán sau huấn luyện đầy đủ vẫn cần được đánh giá.

## 1 Lý do chọn đề tài

Tương tác giữa các thuốc là một bài toán có ý nghĩa trong nghiên cứu tin sinh học và ứng dụng học máy vào dữ liệu phân tử. Trong phạm vi môn học, bài toán này tạo điều kiện thực hành toàn bộ quy trình Deep Learning: biểu diễn dữ liệu phi ảnh, xây dựng mạng nhiều tầng, xử lý mất cân bằng lớp, huấn luyện và đánh giá mô hình.

Em chọn hướng sử dụng cấu trúc phân tử vì dữ liệu hiện có chứa SMILES của từng thuốc và nhãn tương tác của cặp thuốc. Fingerprint giúp chuyển cấu trúc phân tử thành đầu vào số có kích thước cố định để triển khai thực nghiệm. Sản phẩm hướng đến một mô hình nghiên cứu và chương trình minh họa dự đoán, không phải công cụ quyết định dùng thuốc trong thực tế.

## 2 Bài toán và mục tiêu

**Phát biểu bài toán:** với cặp thuốc có thứ tự A và B, mô hình nhận cấu trúc phân tử của A và B, sau đó dự đoán một trong 86 lớp tương tác. Phạm vi hiện tại là phân loại đơn nhãn trên các cặp đã có nhãn tương tác; mô hình không giải quyết riêng câu hỏi một cặp bất kỳ có tương tác hay không.

Ba mục tiêu thực hiện gồm:

1. Xây dựng quy trình chuẩn hóa SMILES, kiểm tra nhãn và chia dữ liệu không trùng cặp cùng thứ tự giữa các tập.
2. Huấn luyện CNN–BiLSTM và so sánh với MLP sử dụng cùng đầu vào fingerprint.
3. Đánh giá tổng thể và theo từng lớp, phân tích các lớp dễ nhầm và lưu mô hình để dự đoán lại mà không cần huấn luyện mỗi lần.

**Câu hỏi thực nghiệm chính:** trên biểu diễn fingerprint hiện tại, CNN–BiLSTM có cải thiện Macro F1 so với MLP hay không, và mức cải thiện có phù hợp với chi phí huấn luyện tăng thêm hay không?

## 3 Dữ liệu và cách xử lý

Dữ liệu đầu vào hiện được lưu trong `SMILES_DDI_filtered.csv`, gồm ba cột: `Smiles1`, `Smiles2` và `Label`. Nhãn nguồn có giá trị từ 1 đến 86; khi huấn luyện, nhãn được chuyển về 0 đến 85. Ý nghĩa bằng lời của từng nhãn và nguồn công bố gốc của bộ dữ liệu cần được đối chiếu, bổ sung vào báo cáo cuối kỳ; hiện chưa tự gán tên loại tương tác cho các mã nhãn.

Quy trình xử lý gồm kiểm tra SMILES hợp lệ, canonical hóa cấu trúc, loại bản ghi trùng và kiểm tra mâu thuẫn nhãn. Dữ liệu hiện có 578 dòng thuộc 289 cặp thuốc cùng thứ tự nhưng mang nhiều nhãn. Các trường hợp này đã được xuất riêng để kiểm tra và không đưa vào thực nghiệm đơn nhãn hiện tại. Nếu xác minh đây là các tương tác đồng thời hợp lệ, hướng phát triển thích hợp sẽ là bài toán đa nhãn thay vì chọn một nhãn tùy ý.

Sau bước xử lý, dữ liệu thực nghiệm còn **191.264 mẫu**, được chia có phân tầng theo nhãn với seed 42:

| Tập dữ liệu | Số mẫu | Tỷ lệ mục tiêu | Số lớp |
|---|---:|---:|---:|
| Huấn luyện | 122.408 | 64% | 86 |
| Validation | 30.603 | 16% | 86 |
| Kiểm tra | 38.253 | 20% | 86 |

Kiểm tra trên các cặp SMILES đã chuẩn hóa cho thấy không có cặp cùng thứ tự bị trùng giữa ba tập. Tuy nhiên, một thuốc có thể xuất hiện trong nhiều tập và cặp đảo chiều có thể nằm ở tập khác nhau. Vì vậy, đây là đánh giá trên các cặp mới trong bối cảnh các thuốc có thể đã được quan sát, chưa phải đánh giá khả năng xử lý thuốc hoàn toàn mới.

Dữ liệu mất cân bằng mạnh: lớp lớn nhất trong train có 38.987 mẫu, lớp nhỏ nhất có 4 mẫu. Một số lớp chỉ có một mẫu ở validation hoặc test. Đây là lý do cần dùng trọng số lớp và thận trọng khi diễn giải chỉ số của từng lớp hiếm.

## 4 Phương pháp Deep Learning đề xuất

### 4.1 Biểu diễn đầu vào

Mỗi SMILES được RDKit chuyển thành fingerprint nhị phân dài 2.048 bit với cấu hình RDKit fingerprint `maxPath=5`. Hai fingerprint được ghép theo đúng thứ tự A rồi B để tạo vector 4.096 chiều. Đầu vào CNN được đổi hình dạng thành `(4096, 1)`. RDKit hỗ trợ đọc cấu trúc từ SMILES và tạo fingerprint; cấu hình cụ thể trong đề tài được giữ thống nhất giữa train và dự đoán [1].

### 4.2 Kiến trúc mô hình chính

Luồng xử lý của mô hình:

```text
SMILES A và SMILES B
        ↓
Chuẩn hóa cấu trúc và tạo hai fingerprint 2048 bit
        ↓
Ghép vector 4096 chiều
        ↓
Bốn khối Conv1D
        ↓
BiLSTM 128 units mỗi chiều
        ↓
BiLSTM 96 units mỗi chiều
        ↓
Dense 256 và Softmax 86 lớp
```

Bốn lớp Conv1D sử dụng lần lượt 16, 32, 48 và 64 filters, kernel size bằng 3, kết hợp Batch Normalization. Ba khối đầu có MaxPooling và Dropout. Lớp BiLSTM đầu trả về chuỗi đặc trưng để truyền sang lớp BiLSTM tiếp theo. Dense 256 sử dụng ReLU và L2 regularization; lớp cuối dùng Softmax. TensorFlow cung cấp lớp Bidirectional để bọc mạng hồi tiếp theo hai chiều [2].

CNN được dùng để thử học các mẫu cục bộ trên vector đầu vào; BiLSTM được dùng để thử tổng hợp đặc trưng theo hai chiều của biểu diễn sau CNN. Tuy nhiên, vị trí các bit fingerprint là kết quả mã hóa và băm đặc trưng phân tử, không tương đương thứ tự nguyên tử hay chuỗi SMILES. Do đó, lợi ích của CNN–BiLSTM là một giả thuyết cần kiểm chứng, không phải kết luận sẵn có về cấu trúc hóa học.

### 4.3 Mô hình đối chiếu

MLP nhận trực tiếp vector 4.096 chiều, đi qua Dense 512, Dropout, Dense 256, Dropout và Softmax 86 lớp. Cả MLP và CNN–BiLSTM đều là mạng học sâu. MLP đóng vai trò mô hình đối chiếu để kiểm tra liệu kiến trúc phức tạp hơn có thực sự mang lại lợi ích trên cùng fingerprint.

## 5 Kế hoạch huấn luyện và đánh giá

Cấu hình khởi đầu gồm Adam với learning rate 0,000173, batch size 32 và tối đa 50 epoch. Hàm mất mát là sparse categorical cross entropy. Trọng số lớp chỉ được tính từ tập train để giảm ảnh hưởng của mất cân bằng. Dropout, L2 regularization, EarlyStopping và giảm learning rate theo validation loss được dùng để hỗ trợ kiểm soát quá khớp.

Checkpoint được chọn theo validation loss. Trong mỗi cấu hình huấn luyện, test chỉ được đọc để đánh giá sau khi đã chọn checkpoint. Các điều chỉnh tiếp theo và lựa chọn giữa hai kiến trúc cần dựa trên validation, tránh dùng test lặp lại để chọn mô hình. Nếu thời gian cho phép, thực nghiệm nên chạy thêm các seed khác để quan sát độ dao động của kết quả.

Các chỉ số sẽ báo cáo gồm Accuracy, Precision, Recall, Macro F1, Weighted F1 và Micro F1, kèm chỉ số từng lớp và ma trận nhầm lẫn. **Macro F1 là chỉ số ưu tiên** vì cho mỗi lớp mức đóng góp ngang nhau; Accuracy có thể bị chi phối bởi lớp phổ biến. Trong bài toán đơn nhãn này, Micro F1 bằng Accuracy nên không được xem là bằng chứng độc lập về chất lượng mô hình.

Việc so sánh sẽ ghi nhận cả thời gian huấn luyện và tài nguyên sử dụng. Nếu CNN–BiLSTM không cải thiện Macro F1 so với MLP, kết quả đó vẫn có ý nghĩa: kiến trúc phức tạp chưa chắc phù hợp hơn với biểu diễn đầu vào hiện tại.

## 6 Tiến độ hiện tại và công việc tiếp theo

| Hạng mục | Trạng thái |
|---|---|
| Chuẩn hóa dữ liệu và chia lại các tập | Đã thực hiện |
| Kiểm tra trùng cặp cùng thứ tự giữa các tập | Đạt với 0 cặp trùng |
| Xây dựng CNN–BiLSTM và MLP | Đã triển khai |
| Kiểm thử tự động | 8 kiểm tra đạt |
| Chạy thử train và đánh giá hai kiến trúc | Đã hoàn tất ở chế độ smoke |
| Huấn luyện đầy đủ và kết luận chất lượng | Chưa có kết quả xác nhận |
| Xác minh nguồn dữ liệu và diễn giải 86 nhãn | Cần bổ sung |

Các lượt smoke đã chạy 1 epoch với 1.000 mẫu train, 300 mẫu validation và 300 mẫu test để kiểm tra chương trình có thể huấn luyện, lưu và đọc lại checkpoint, xuất chỉ số và dự đoán từ SMILES. **Đây là kiểm tra chức năng, không phải kết quả thực nghiệm cuối cùng.**

Công việc tiếp theo là xác minh mô tả bộ dữ liệu, huấn luyện đầy đủ hai mô hình, so sánh bằng validation, sau đó báo cáo kết quả test của mô hình được chọn. Huấn luyện đầy đủ dự kiến thực hiện trên môi trường có GPU nếu sẵn có; có thể giảm batch size khi bộ nhớ hạn chế. Sau khi có checkpoint, việc dự đoán chỉ cần đọc model đã lưu.

## 7 Sản phẩm dự kiến và phạm vi hoàn thành

Sản phẩm chính gồm mã nguồn xử lý dữ liệu, mô hình đã huấn luyện, biểu đồ loss và accuracy, bảng chỉ số đánh giá, ma trận nhầm lẫn, bảng so sánh CNN–BiLSTM với MLP và một số ví dụ dự đoán. Chương trình minh họa nhận hai SMILES rồi trả các mã lớp có xác suất Softmax cao nhất.

Giao diện web là phần mở rộng nếu thời gian cho phép; trọng tâm môn học là mô hình và chất lượng thực nghiệm. Chưa đặt trước một ngưỡng Accuracy hay khẳng định vượt trội khi chưa có kết quả. Các hạn chế cần trình bày gồm mất cân bằng lớp, mẫu hiếm, cách ly các cặp nhiều nhãn, giới hạn của fingerprint và việc split hiện tại chưa đánh giá thuốc hoàn toàn mới.

Nếu tiếp tục phát triển sau phạm vi môn học, có thể xem xét học trực tiếp trên chuỗi SMILES hoặc dùng mạng nơ-ron đồ thị để khai thác biểu diễn phân tử khác. Đây là hướng mở rộng, chưa phải thành phần đã triển khai.

## 8 Nội dung trình bày miệng khoảng hai phút

“Em xin đề xuất đề tài phân loại tương tác giữa hai thuốc từ cấu trúc phân tử bằng mô hình CNN kết hợp BiLSTM. Bài toán nhận đầu vào là hai chuỗi SMILES và dự đoán một trong 86 lớp tương tác đã có nhãn trong bộ dữ liệu.

Em chuyển mỗi thuốc thành fingerprint RDKit 2.048 bit, ghép thành vector 4.096 chiều, sau đó đưa qua CNN, BiLSTM và lớp Softmax. Em đồng thời xây dựng MLP để so sánh vì fingerprint là vector bit, nên chưa thể giả định CNN–BiLSTM sẽ tốt hơn.

Sau khi chuẩn hóa và cách ly các trường hợp mâu thuẫn nhãn, dữ liệu còn khoảng 191 nghìn mẫu. Em chia train, validation và test theo tỷ lệ 64, 16 và 20 phần trăm, đồng thời kiểm tra để không trùng cặp cùng thứ tự giữa các tập. Do dữ liệu mất cân bằng, em ưu tiên Macro F1 bên cạnh Accuracy và phân tích ma trận nhầm lẫn.

Hiện em đã xây dựng pipeline và chạy thử thành công hai mô hình, nhưng chưa có kết quả huấn luyện đầy đủ để kết luận chất lượng. Bước tiếp theo là train đầy đủ, lựa chọn mô hình bằng validation và tổng hợp kết quả test. Sản phẩm chính là model đã lưu, báo cáo thực nghiệm và chương trình minh họa dự đoán; giao diện là phần mở rộng nếu còn thời gian.”

## 9 Một số câu hỏi có thể gặp

**Vì sao đây là Deep Learning khi fingerprint được RDKit tạo sẵn?**  
Fingerprint là bước biểu diễn dữ liệu. Các trọng số trong mạng nhiều tầng CNN–BiLSTM hoặc MLP vẫn được học từ dữ liệu bằng lan truyền ngược; vì vậy phần phân loại vẫn theo hướng Deep Learning.

**Vì sao cần MLP?**  
Để có mô hình đối chiếu dùng cùng dữ liệu, từ đó đánh giá lợi ích và chi phí của CNN–BiLSTM thay vì chỉ báo cáo một kiến trúc.

**Model có xác định thuốc dùng chung an toàn không?**  
Không. Phạm vi hiện tại là phân loại các lớp tương tác trong bộ dữ liệu; chưa có lớp “không tương tác”, chưa xác minh đầy đủ ý nghĩa các nhãn và chưa được kiểm định để sử dụng thực tế.

**Tính mới nằm ở đâu?**  
Đây là đề tài ứng dụng trong phạm vi môn học, không tuyên bố kiến trúc mới. Giá trị thực hiện nằm ở pipeline có thể tái lập, kiểm tra dữ liệu và so sánh mô hình có phân tích giới hạn.

## Tài liệu tham khảo

[1] RDKit. Getting Started with the RDKit in Python. https://www.rdkit.org/docs/GettingStartedInPython.html

[2] TensorFlow. tf.keras.layers.Bidirectional. https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional

Các số liệu dữ liệu và tiến độ triển khai được đối chiếu từ `data/processed/metadata.json`, `VALIDATION.md`, `preprocess.py`, `model.py` và `predict.py` trong project. Tài liệu tham khảo trên mô tả công cụ và API; không thay thế trích dẫn nguồn gốc bộ dữ liệu cần bổ sung.
