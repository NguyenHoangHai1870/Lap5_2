# === NHIỆM VỤ 4: Embedding (scratch) + LSTM ===
lstm_scr = Sequential([
    Embedding(input_dim=vocab_size, output_dim=100, input_length=max_len),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(num_classes, activation='softmax')
])
lstm_scr.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
_ = lstm_scr.fit(X_train_pad, y_train, validation_data=(X_val_pad, y_val),
                 epochs=30, batch_size=64, callbacks=[es], verbose=0)

y_pred_scr = np.argmax(lstm_scr.predict(X_test_pad, verbose=0), axis=1)
print("=== Classification report: Emb(scratch) + LSTM ===")
print(classification_report(y_test, y_pred_scr, target_names=le.classes_, digits=4))
f1_scr = f1_score(y_test, y_pred_scr, average="macro")
print("Macro-F1 (test):", f1_scr)
