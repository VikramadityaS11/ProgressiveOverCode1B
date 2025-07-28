import pandas as pd
import json

import pandas as pd
import json


import re

def clean_text(text):
    if not isinstance(text, str):
        return text
    # Replace newline characters with space
    text = text.replace('\n', ' ')
    
    # Replace Unicode quotes with ASCII equivalents
    replacements = {
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote
        '\u201c': '"',  # Left double quote
        '\u201d': '"',  # Right double quote
        '\u2013': '-',  # En dash
        '\u2014': '-',  # Em dash
        '\u2026': '...',  # Ellipsis
        '\xa0': ' ',    # Non-breaking space
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_sections_between_headings(df_all, heading_df):
    heading_df = heading_df.sort_values(["page_num", "y"]).reset_index(drop=True)
    df_all = df_all.sort_values(["page_num", "y"]).reset_index(drop=True)

    sections = []

    for i, row in heading_df.iterrows():
        start_page = row["page_num"]
        start_y = row["y"]
        start_idx = df_all[(df_all["page_num"] == start_page) & (df_all["y"] >= start_y)].index.min()

        if pd.isna(start_idx):
            continue  # Skip headings that don't match any text line

        if i + 1 < len(heading_df):
            next_row = heading_df.iloc[i + 1]
            end_page = next_row["page_num"]
            end_y = next_row["y"]
            end_idx = df_all[(df_all["page_num"] == end_page) & (df_all["y"] >= end_y)].index.min()

            if pd.isna(end_idx):
                end_idx = len(df_all)
        else:
            end_idx = len(df_all)

        content_rows = df_all.iloc[int(start_idx): int(end_idx)]
        content_text = "\n".join(content_rows["text"].astype(str).tolist()).strip()
        if not content_text:
            continue
        full_content = f"{row['text']}\n{content_text}"


        sections.append({
            "heading": row["text"],
            "page": int(row["page_num"]) + 1,
            "content": full_content
        })

    return sections

