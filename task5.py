# === NHIỆM VỤ 5: So sánh định lượng + Phân tích định tính ===
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report

loss_avg, _ = model_avg.evaluate(X_test_avg, y_test, verbose=0)
loss_pre, _ = lstm_pre.evaluate(X_test_pad, y_test, verbose=0)
loss_scr, _ = lstm_scr.evaluate(X_test_pad, y_test, verbose=0)

summary = pd.DataFrame({
    "Pipeline": [
        "TF-IDF + Logistic Regression",
        "Word2Vec (Avg) + Dense",
        "Embedding (Pre-trained) + LSTM",
        "Embedding (Scratch) + LSTM"
    ],
    "F1-macro (test)": [f1_lr, f1_avg, f1_pre, f1_scr],
    "Test Loss": [None, loss_avg, loss_pre, loss_scr]
}).sort_values("F1-macro (test)", ascending=False).reset_index(drop=True)

display(summary)

hard_texts = [
    "can you remind me to not call my mom",
    "is it going to be sunny or rainy tomorrow",
    "find a flight from new york to london but not through paris"
]

def predict_all(texts):
    pred_lr = tfidf_lr_pipeline.predict(texts)
    Xavg = np.vstack([sentence_to_avg_vector(t, w2v, w2v_dim) for t in texts])
    pred_avg = np.argmax(model_avg.predict(Xavg, verbose=0), axis=1)
    seqs = tokenizer.texts_to_sequences(texts)
    Xpad = pad_sequences(seqs, maxlen=max_len, padding='post', truncating='post')
    pred_pre = np.argmax(lstm_pre.predict(Xpad, verbose=0), axis=1)
    pred_scr = np.argmax(lstm_scr.predict(Xpad, verbose=0), axis=1)
    return pred_lr, pred_avg, pred_pre, pred_scr

pred_lr, pred_avg, pred_pre, pred_scr = predict_all(hard_texts)

true_labels = []
for t in hard_texts:
    match = df_test[df_test["text"].str.lower() == t.lower()]
    if len(match) > 0:
        true_labels.append(le.transform(match["intent"])[0])
    else:
        true_labels.append(None)

rows = []
for i, text in enumerate(hard_texts):
    y_true = true_labels[i]
    row = {
        "text": text,
        "True Intent": le.classes_[y_true] if y_true is not None else "N/A",
        "TF-IDF + LR": le.classes_[pred_lr[i]],
        "W2V-Avg + Dense": le.classes_[pred_avg[i]],
        "LSTM (pre)": le.classes_[pred_pre[i]],
        "LSTM (scratch)": le.classes_[pred_scr[i]],
    }
    if y_true is not None:
        row["✓ LR"] = "✓" if pred_lr[i] == y_true else "✗"
        row["✓ W2V"] = "✓" if pred_avg[i] == y_true else "✗"
        row["✓ LSTM-pre"] = "✓" if pred_pre[i] == y_true else "✗"
        row["✓ LSTM-scr"] = "✓" if pred_scr[i] == y_true else "✗"
    rows.append(row)

qual_df = pd.DataFrame(rows)
display(qual_df)

print("\n Nhận xét:")
print("- Các câu có phủ định ('not') hoặc mệnh đề phụ ('or', 'but not through ...') thường khiến mô hình TF-IDF và W2V trung bình dự đoán sai, "
      "vì chúng không nắm được thứ tự và phạm vi của phủ định.")
print("- LSTM (đặc biệt bản pre-trained) có xu hướng hiểu ngữ cảnh tốt hơn, vì trạng thái ẩn theo chuỗi giúp mô hình nhận diện được mối quan hệ ngữ pháp và ý phủ định.")
print("- Nếu LSTM (pre-trained) chính xác hơn bản học từ đầu, điều đó chứng tỏ embedding Word2Vec cung cấp ngữ nghĩa ban đầu hữu ích, giúp mô hình hội tụ nhanh và tổng quát tốt hơn.")
