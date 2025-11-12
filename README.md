1. Mục tiêu
Mục tiêu của bài thực hành là tìm hiểu, xây dựng và so sánh các mô hình phân loại văn bản từ truyền thống đến hiện đại, qua đó hiểu rõ vai trò của việc mô hình hóa ngữ cảnh trong ngôn ngữ tự nhiên. Cụ thể, bốn mô hình được triển khai và đánh giá: TF-IDF + Logistic Regression (Baseline 1) Word2Vec (trung bình) + Dense Layer (Baseline 2) Embedding Layer (Pre-trained) + LSTM Embedding Layer (Học từ đầu) + LSTM

2. Hạn chế của các mô hình truyền thống
2.1 Bag-of-Words / TF-IDF

Mất thông tin thứ tự từ: Hai câu “not good” và “good not” có cùng vector TF-IDF, dù ý nghĩa hoàn toàn ngược nhau. Không nắm ngữ cảnh phủ định hay phụ thuộc xa: TF-IDF xem từ độc lập, không hiểu quan hệ cú pháp hay ngữ pháp. Hiệu quả với câu ngắn hoặc từ khóa rõ, nhưng kém với câu tự nhiên, hội thoại. 2.2 Word2Vec (Trung bình)

Cải thiện ngữ nghĩa từ: Từ được biểu diễn bằng vector dense có quan hệ ngữ nghĩa (“king – man + woman ≈ queen”). Tuy nhiên, mất cấu trúc chuỗi khi lấy trung bình → “not good” và “good” có vector gần nhau, gây nhầm ý nghĩa phủ định. Phù hợp hơn TF-IDF cho dữ liệu ngắn, nhưng vẫn chưa “hiểu” câu.

3. Kiến trúc và Luồng pipeline RNN/LSTM
3.1 Pipeline tổng thể

Văn bản → Tokenizer → Sequence (IDs) → Padding → Embedding → LSTM → Dense (Softmax)
Tokenizer: Tạo từ điển và chuyển câu thành chuỗi chỉ số. Padding: Căn độ dài chuỗi bằng nhau (ví dụ max_len = 50). Embedding: Biểu diễn chỉ số từ bằng vector dense. LSTM: Xử lý chuỗi theo thời gian, lưu “trạng thái ngữ cảnh” giúp nắm phủ định và phụ thuộc xa. Dense + Softmax: Phân loại câu theo ý định (intent).

3.2 Hai kiểu Embedding trong bài

Pre-trained: Khởi tạo từ trọng số Word2Vec đã huấn luyện → học nhanh, ngữ nghĩa tốt. Scratch: Học từ đầu trong quá trình training → linh hoạt hơn với domain mới.

4. Bảng so sánh kết quả định lượng
```
| *Pipeline*                   | *F1-macro (test)* | *Test Loss* |
| ------------------------------ | ------------------- | ------------- |
| TF-IDF + Logistic Regression   | *0.835298*        | N/A           |
| Word2Vec (Avg) + Dense         | 0.149018            | 3.032118      |
| Embedding (Scratch) + LSTM     | 0.149525            | 2.941058      |
| Embedding (Pre-trained) + LSTM | 0.036167            | 3.603505      |
```
Nhận xét định lượng: Mô hình TF-IDF + LR đạt kết quả cao nhất về F1 (≈0.83). Cả ba mô hình dùng neural network (W2V + Dense, LSTM) có F1 rất thấp (≈0.14–0.03).

Lý do có thể:

Dữ liệu huấn luyện ít hoặc chưa được token hóa đồng bộ. Các mô hình LSTM cần nhiều epoch và learning rate phù hợp để hội tụ. Embedding pre-trained không tương thích với vocabulary của tập dữ liệu HWU.

Kết luận tạm thời: Baseline truyền thống (TF-IDF + LR) vẫn hoạt động tốt và ổn định, trong khi các mô hình neural cần tinh chỉnh thêm.

5️. Phân tích định tính (Qualitative)
```
| *Câu kiểm thử*                                            | *Intent thật* | *TF-IDF + LR* | *W2V-Avg + Dense* | *LSTM (Pre-trained)* | *LSTM (Scratch)* |
| ----------------------------------------------------------- | --------------- | --------------- | ------------------- | ---------------------- | ------------------ |
| can you remind me to not call my mom                        | N/A             | calendar_set    | general_joke        | general_dontcare       | email_sendemail    |
| is it going to be sunny or rainy tomorrow                   | weather_query   | weather_query   | calendar_query      | lists_createoradd      | social_post        |
| find a flight from new york to london but not through paris | N/A             | general_negate  | transport_query     | alarm_set              | cooking_recipe     |
```
Phân tích:

Các mô hình neural đều dự đoán sai, trong khi TF-IDF ít nhất cho ra nhãn “weather_query” đúng cho câu 2. Câu phủ định (“not call”, “not through”) → các mô hình neural không học được phạm vi phủ định. Câu ghép (“sunny or rainy”) → cần hiểu cấu trúc song song, nhưng embedding không đủ mạnh. LSTM pre-trained chưa vượt trội do embedding chưa khớp domain hoặc bị giới hạn số epoch.

6. Nhận xét tổng quát
```
| *Mô hình*            | *Ưu điểm*                                               | *Nhược điểm*                                        |
| ---------------------- | --------------------------------------------------------- | ----------------------------------------------------- |
| *TF-IDF + LR*        | Dễ triển khai, ổn định, không overfit, phù hợp dữ liệu ít | Không hiểu ngữ cảnh, mất thứ tự từ                    |
| *W2V-Avg + Dense*    | Có vector ngữ nghĩa từ                                    | Mất cấu trúc chuỗi, F1 thấp nếu không fine-tune       |
| *LSTM (Pre-trained)* | Lý thuyết mạnh, mô hình hóa chuỗi và phụ thuộc xa         | Cần dữ liệu lớn, dễ underfit nếu embedding không khớp |
| *LSTM (Scratch)*     | Linh hoạt với domain mới                                  | Học chậm, cần huấn luyện dài và regularization tốt    |
```
7. Kết luận
Trong thực nghiệm này, TF-IDF + Logistic Regression vẫn hoạt động hiệu quả nhất (F1 ≈ 0.83). Các mô hình neural chưa thể hiện ưu thế do giới hạn huấn luyện (số epoch, embedding không tương thích). Tuy nhiên, về mặt lý thuyết, LSTM vẫn vượt trội khi xử lý ngữ cảnh, phủ định và phụ thuộc xa — nếu được huấn luyện đúng và có đủ dữ liệu.
