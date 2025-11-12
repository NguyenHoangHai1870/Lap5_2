# === NHIỆM VỤ 2: Word2Vec Avg + Dense ===
import numpy as np
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

sentences = [str(s).split() for s in df_train["text"]]
w2v_dim = 100
w2v = Word2Vec(sentences=sentences, vector_size=w2v_dim, window=5, min_count=1, workers=4, seed=42)

def sentence_to_avg_vector(text, model, dim=100):
    toks = str(text).split()
    vecs = [model.wv[t] for t in toks if t in model.wv]
    if not vecs:
        return np.zeros(dim, dtype="float32")
    return np.mean(vecs, axis=0).astype("float32")

def to_matrix(texts, model, dim=100):
    return np.vstack([sentence_to_avg_vector(t, model, dim) for t in texts])

X_train_avg = to_matrix(df_train["text"], w2v, w2v_dim)
X_val_avg   = to_matrix(df_val["text"],   w2v, w2v_dim)
X_test_avg  = to_matrix(df_test["text"],  w2v, w2v_dim)

model_avg = Sequential([
    Dense(128, activation='relu', input_shape=(w2v_dim,)),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
model_avg.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
_ = model_avg.fit(X_train_avg, y_train, validation_data=(X_val_avg, y_val),
                  epochs=50, batch_size=64, callbacks=[es], verbose=0)

y_pred_avg = np.argmax(model_avg.predict(X_test_avg, verbose=0), axis=1)
print("=== Classification report: W2V-Avg + Dense ===")
print(classification_report(y_test, y_pred_avg, target_names=le.classes_, digits=4))
from sklearn.metrics import f1_score
f1_avg = f1_score(y_test, y_pred_avg, average="macro")
print("Macro-F1 (test):", f1_avg)
