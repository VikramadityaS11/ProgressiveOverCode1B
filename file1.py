
from pathlib import Path
import json
from collections import defaultdict, Counter
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix




ann_dir_train = Path("base_dataset/train/annotations")    # from adobe/src/
pdf_dir_train = Path("base_dataset/train/pdfs")
ann_dir_test = Path("base_dataset/test/annotations")
pdf_dir_test = Path("base_dataset/test/pdfs")    




regions = defaultdict(list) # key: category_id, value: list of region dicts
cat2id = {"Title": 1, "Section-header": 2, "Text": 3}


# Helper to load and process annotations from any annotation directory
def load_regions_from_dir(ann_dir):
    for ann_path in ann_dir.glob("*.json"):
        with open(ann_path, encoding="utf8") as f:
            page = json.load(f)
        page_width = page["metadata"]["original_width"]
        page_height = page["metadata"]["original_height"]
        for region in page["form"]:
            cat = region["category"]
            if cat in cat2id:
                region["pdf_file"] = ann_path.stem + ".pdf"
                region["label_id"] = cat2id[cat]
                region["page_width"] = page_width
                region["page_height"] = page_height
                regions[cat2id[cat]].append(region)

# Load from both train and test splits
load_regions_from_dir(ann_dir_train)
load_regions_from_dir(ann_dir_test)

# Example: print region counts
# To access all 'Title' regions: regions[1]
# To access all 'Section-header' regions: regions[2]
# To access all 'Text' regions: regions[3]

min_n = min(len(regions[1]), len(regions[2]), len(regions[3]))



balanced = []
for lbl in [1, 2, 3]:
    balanced.extend(random.sample(regions[lbl], min_n))
random.shuffle(balanced)



def region_to_features(region):
    # --- Basic text features ---
    text = region["text"]
    word_count = len(text.split())
    char_count = len(text)
    upper_ratio = sum(1 for c in text if c.isupper()) / (len(text) or 1)
    digit_ratio = sum(1 for c in text if c.isdigit()) / (len(text) or 1)
    punct_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / (len(text) or 1)
    first_word_upper = int(text.split()[0].isupper()) if text.split() else 0
    is_all_upper = int(text.isupper())

    # --- Font/style features ---
    font = region.get("font", {})
    font_size = font.get("size", 0)
    font_name = font.get("name", "")
    bold = int("Bold" in font_name)
    italic = int("Italic" in font_name or "Oblique" in font_name)

    # --- Geometry & layout features ---
    x, y, w, h = region["box"]
    area = w * h
    aspect_ratio = w / h if h != 0 else 0

    # Optionally, normalization to page size:
    norm_x = x / region.get("page_width",None)
    norm_y = y / region.get("page_height",None)
    norm_w = w / region.get("page_width",None)
    norm_h = h / region.get("page_height",None)

    return {
        "text": text,                   # for debugging/inspection
        "word_count": word_count,
        "char_count": char_count,
        "upper_ratio": upper_ratio,
        "digit_ratio": digit_ratio,
        "punct_ratio": punct_ratio,
        "first_word_upper": first_word_upper,
        "is_all_upper": is_all_upper,
        "font_size": font_size,
        "bold": bold,
        "italic": italic,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "area": area,
        "aspect_ratio": aspect_ratio,
        "norm_x": norm_x,
        "norm_y": norm_y,
        "norm_w": norm_w,
        "norm_h": norm_h,
        "label": region["label_id"],
    }


# Build the feature table
features = [region_to_features(r) for r in balanced]
df = pd.DataFrame(features)

label_map = {1: 0, 2: 1, 3: 2}
df['label'] = df['label'].map(label_map)
y = df['label']


X = df.drop(['text', 'label'], axis=1)
y = df['label']





X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'seed': 42
}

model = lgb.train(params, train_data, valid_sets=[test_data], callbacks=[lgb.early_stopping(10)])
model.save_model('lightgbm_model.txt')




y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = np.argmax(y_pred_proba, axis=1)
# Evaluate with classification_report, confusion_matrix, etc.



print(classification_report(
    y_test, y_pred,
    target_names=['Title', 'Section-header', 'Text']
))

print(confusion_matrix(y_test, y_pred))

