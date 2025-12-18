# === NHIỆM VỤ 1: TF-IDF + Logistic Regression ===
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, f1_score

tfidf_lr_pipeline = make_pipeline(
    TfidfVectorizer(max_features=5000),
    LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)
)

tfidf_lr_pipeline.fit(df_train["text"], y_train)
y_pred_lr = tfidf_lr_pipeline.predict(df_test["text"])

print("=== Classification report: TF-IDF + LR ===")
print(classification_report(y_test, y_pred_lr, target_names=le.classes_, digits=4))
f1_lr = f1_score(y_test, y_pred_lr, average="macro")
print("Macro-F1 (test):", f1_lr)
