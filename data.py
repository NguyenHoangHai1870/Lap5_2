
import os, tarfile
import pandas as pd
from sklearn.preprocessing import LabelEncoder

TAR_PATH = "data/hwu.tar.gz"
if os.path.exists(TAR_PATH):
    with tarfile.open(TAR_PATH, "r:gz") as tar:
        tar.extractall("data")
    print(" Đã giải nén:", TAR_PATH)
else:
    try:
        from google.colab import files
        print(" Upload hwu.tar.gz HOẶC train/val/test (.csv)")
        up = files.upload()
        os.makedirs("data", exist_ok=True)
        for name, content in up.items():
            open(os.path.join("data", name), "wb").write(content)
        for name in up.keys():
            if name.endswith(".tar.gz") or name.endswith(".tgz"):
                with tarfile.open(os.path.join("data", name), "r:gz") as tar:
                    tar.extractall("data")
                print(" Đã giải nén:", name)
    except Exception as e:
        print(" Không chạy trong Colab hoặc upload thất bại:", e)

data_dir = "data/hwu" if os.path.isdir("data/hwu") else "data"
print(" data_dir =", data_dir, "| Files:", os.listdir(data_dir))

train_path = os.path.join(data_dir, "train.csv")
val_path   = os.path.join(data_dir, "val.csv")
test_path  = os.path.join(data_dir, "test.csv")
assert os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path), \
       "Không tìm thấy train.csv/val.csv/test.csv trong " + data_dir

df_train = pd.read_csv(train_path)
df_val   = pd.read_csv(val_path)
df_test  = pd.read_csv(test_path)

def normalize_cols(df):
    cols = {c.lower(): c for c in df.columns}
    text_col = None
    for k in ["text","utterance","sentence","query","content"]:
        if k in cols: text_col = cols[k]; break
    intent_col = None
    for k in ["intent","label","category","class","target"]:
        if k in cols: intent_col = cols[k]; break
    if text_col is None: text_col = df.columns[0]
    if intent_col is None: intent_col = df.columns[1]
    return df.rename(columns={text_col:"text", intent_col:"intent"})[["text","intent"]]

df_train = normalize_cols(df_train)
df_val   = normalize_cols(df_val)
df_test  = normalize_cols(df_test)

print("Train shape:", df_train.shape)
print("Validation shape:", df_val.shape)
print("Test shape:", df_test.shape)
display(df_train.head())

le = LabelEncoder()
le.fit(pd.concat([df_train["intent"], df_val["intent"], df_test["intent"]], axis=0))

y_train = le.transform(df_train["intent"])
y_val   = le.transform(df_val["intent"])
y_test  = le.transform(df_test["intent"])
num_classes = len(le.classes_)
print(" num_classes:", num_classes, "| ví dụ nhãn:", list(le.classes_)[:10], "…")
