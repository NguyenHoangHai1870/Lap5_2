# === NHIỆM VỤ 3: Embedding (pre-trained) + LSTM ===
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM
from tensorflow.keras.models import Sequential

max_words = 20000
oov_tok = "<UNK>"
tokenizer = Tokenizer(num_words=max_words, oov_token=oov_tok)
tokenizer.fit_on_texts(df_train["text"])

def to_pad(texts, tok, max_len=50):
    seqs = tok.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=max_len, padding='post', truncating='post')

max_len = 50
X_train_pad = to_pad(df_train["text"], tokenizer, max_len)
X_val_pad   = to_pad(df_val["text"],   tokenizer, max_len)
X_test_pad  = to_pad(df_test["text"],  tokenizer, max_len)

vocab_size = min(max_words, len(tokenizer.word_index) + 1)
embedding_dim = w2v_dim

embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype="float32")
for word, idx in tokenizer.word_index.items():
    if idx < vocab_size and word in w2v.wv:
        embedding_matrix[idx] = w2v.wv[word]

lstm_pre = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim,
              weights=[embedding_matrix], input_length=max_len, trainable=False),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(num_classes, activation='softmax')
])
lstm_pre.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
_ = lstm_pre.fit(X_train_pad, y_train, validation_data=(X_val_pad, y_val),
                 epochs=30, batch_size=64, callbacks=[es], verbose=0)

y_pred_pre = np.argmax(lstm_pre.predict(X_test_pad, verbose=0), axis=1)
print("=== Classification report: Emb(pretrained) + LSTM ===")
print(classification_report(y_test, y_pred_pre, target_names=le.classes_, digits=4))
f1_pre = f1_score(y_test, y_pred_pre, average="macro")
print("Macro-F1 (test):", f1_pre)
