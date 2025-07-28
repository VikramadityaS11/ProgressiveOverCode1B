from file2 import extract_region_features_grouped, assign_heading_levels
from extract_sections import extract_sections_between_headings, clean_text
import lightgbm as lgb

import os
import json
import numpy as np

combined_outlines = []
combined_sections = []
def run():
    model = lgb.Booster(model_file='lightgbm_model.txt')

    # Directory containing PDFs
    input_dir = "/Users/vikramadityasharma/Desktop/Adobe-India-Hackathon25/Challenge_1b/Collection 2/PDFs"

    label_map = {0: 'Title', 1: 'Section-header', 2: 'Text'}

    def build_outline_json(header_df):
        if header_df.empty:
            return {"title": "", "outline": []}

        header_df = header_df.sort_values(["page_num", "y"], ascending=[True, True])
        first_page = header_df[header_df["page_num"] == header_df["page_num"].min()]
        title_row = first_page[first_page["heading_level"] == "title"].head(1)

        if title_row.empty:
            h_rows = first_page[first_page["heading_level"].str.startswith("H")]
            if not h_rows.empty:
                h_rows["level_num"] = h_rows["heading_level"].str.extract(r'H(\d+)').astype(int)
                min_level = h_rows["level_num"].min()
                title_row = h_rows[h_rows["level_num"] == min_level].sort_values("y").head(1)
            else:
                title_row = first_page.head(1)

        title_text = title_row["text"].values[0]
        header_df = header_df[header_df["text"] != title_text]

        outline = [
            {
                "level": row["heading_level"],
                "text": row["text"],
                "page": int(row["page_num"]) + 1
            }
            for _, row in header_df.iterrows()
            if row["heading_level"].startswith("H")
        ]

        return {
            "title": title_text,
            "outline": outline
        }

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".pdf"):
            continue

        filepath = os.path.join(input_dir, filename)
        print(f"Processing: {filename}")

        try:
            df = extract_region_features_grouped(filepath)
            X = df.drop(columns=['text', 'page_width', 'page_height', 'page_num'])
            y_pred_proba = model.predict(X, num_iteration=model.best_iteration)
            y_pred = np.argmax(y_pred_proba, axis=1)
            df['predicted'] = [label_map[i] for i in y_pred]

            header_df = df[df['predicted'].isin(['Title', 'Section-header'])].copy()
            header_df = assign_heading_levels(header_df)

            # Outline
            outline = build_outline_json(header_df)
            outline["source_file"] = filename
            combined_outlines.append(outline)

            # Sections
            text_df = df[df['predicted'] == 'Text'][['text', 'x', 'y', 'page_num']].copy()
            text_df['text'] = text_df['text'].apply(clean_text)

            heading_df = header_df[['text', 'x', 'y', 'page_num', 'heading_level']].copy()
            heading_df = heading_df.dropna(subset=['text', 'page_num'])
            heading_df['text'] = heading_df['text'].apply(clean_text)

            sections = extract_sections_between_headings(text_df, heading_df)
            for section in sections:
                section['source_file'] = filename
            combined_sections.extend(sections)

        except Exception as e:
            print(f"Failed for {filename}: {e}")

if __name__ == "main":
    run()
