import fitz  # PyMuPDF
import pandas as pd
import string
import numpy as np
import lightgbm as lgb
import json
import pdfplumber
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans



def group_lines_into_blocks(lines, vertical_threshold=2.5):
    lines_sorted = sorted(lines, key=lambda l: l['y'])
    blocks = []
    current_block = [lines_sorted[0]]
    for prev, curr in zip(lines_sorted, lines_sorted[1:]):
        if curr['y'] - (prev['y'] + prev['h']) <= vertical_threshold:
            current_block.append(curr)
        else:
            blocks.append(current_block)
            current_block = [curr]
    blocks.append(current_block)
    return blocks

def extract_region_features_grouped(pdf_path):
    doc = fitz.open(pdf_path)
    regions = []

    for page in doc:
        page_number = page.number
        page_width = page.rect.width
        page_height = page.rect.height
        lines = []

        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                spans = line["spans"]
                texts = [s["text"] for s in spans]
                text = "".join(texts).strip()
                if not text:
                    continue
                first_span = spans[0]
                font_size = first_span.get("size", 0)
                font_name = first_span.get("font", "").lower()
                bold = int("bold" in font_name)
                italic = int(("italic" in font_name) or ("oblique" in font_name))
                x0, y0, x1, y1 = line["bbox"]
                w, h = x1 - x0, y1 - y0
                line_obj = {
                    'text': text,
                    'font_size': font_size,
                    'bold': bold,
                    'italic': italic,
                    'x': x0,
                    'y': y0,
                    'w': w,
                    'h': h,
                    'bbox': fitz.Rect(x0, y0, x1, y1),
                    'page_num': page_number
                }
                lines.append(line_obj)

        if not lines:
            continue

        blocks = group_lines_into_blocks(lines)
        for block in blocks:
            texts = [l['text'] for l in block]
            full_text = '\n'.join(texts)
            min_x = min(l['x'] for l in block)
            min_y = min(l['y'] for l in block)
            max_x = max(l['x'] + l['w'] for l in block)
            max_y = max(l['y'] + l['h'] for l in block)

            w = max_x - min_x
            h = max_y - min_y
            font_size = np.mean([l['font_size'] for l in block])
            bold = int(sum(l['bold'] for l in block) > len(block) / 2)
            italic = int(sum(l['italic'] for l in block) > len(block) / 2)

            region = {
                'text': full_text,
                'word_count': len(full_text.split()),
                'char_count': len(full_text),
                'upper_ratio': sum(1 for c in full_text if c.isupper()) / (len(full_text) or 1),
                'digit_ratio': sum(1 for c in full_text if c.isdigit()) / (len(full_text) or 1),
                'punct_ratio': sum(1 for c in full_text if c in string.punctuation) / (len(full_text) or 1),
                'first_word_upper': int(full_text.split()[0].isupper()) if full_text.split() else 0,
                'is_all_upper': int(full_text.isupper()),
                'font_size': font_size,
                'bold': bold,
                'italic': italic,
                'x': min_x,
                'y': min_y,
                'w': w,
                'h': h,
                'area': w * h,
                'aspect_ratio': w / h if h != 0 else 0,
                'norm_x': min_x / page_width if page_width else 0,
                'norm_y': min_y / page_height if page_height else 0,
                'norm_w': w / page_width if page_width else 0,
                'norm_h': h / page_height if page_height else 0,

                'page_height': page_height,
                'page_width': page_width,
                'page_num': page_number

            }
            regions.append(region)

    df = pd.DataFrame(regions)
    return df



def assign_heading_levels(header_df, max_levels=6):
    if header_df.empty:
        return header_df

    features = header_df[["font_size", "norm_x"]].values

    n_samples = len(features)
    n_clusters = min(max_levels, n_samples)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    header_df["cluster_id"] = kmeans.fit_predict(features)


    header_df["text"] = header_df["text"].str.replace("\n", " ", regex=False).str.strip()

    cluster_to_avg_size = header_df.groupby("cluster_id")["font_size"].mean().to_dict()
    sorted_clusters = sorted(cluster_to_avg_size.items(), key=lambda x: -x[1])
    cluster_to_level = {cid: f"H{i+1}" for i, (cid, _) in enumerate(sorted_clusters)}

    header_df["heading_level"] = header_df["cluster_id"].map(cluster_to_level)
    return header_df


