import fitz  # PyMuPDF
import re
from collections import Counter
import statistics

def clean_text(text):
    """Cleans up extracted text."""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def is_title_case(text):
    """Checks if a string is in title case."""
    return text.istitle()

def get_page_stats(page):
    """
    Calculates statistics for a page needed for feature engineering.
    """
    font_sizes = Counter()
    vertical_spaces = []
    last_y1 = 0
    
    lines = []
    blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_SEARCH)["blocks"]
    for block in blocks:
        if block['type'] == 0:
            for line in block["lines"]:
                lines.append(line)
                if last_y1 > 0:
                    space = line['bbox'][1] - last_y1
                    if space > 0 and space < 50: # Ignore large gaps
                        vertical_spaces.append(space)
                last_y1 = line['bbox'][3]
                for span in line["spans"]:
                    font_sizes[round(span['size'])] += 1

    body_font_size = font_sizes.most_common(1)[0][0] if font_sizes else 12.0
    avg_vertical_space = statistics.mean(vertical_spaces) if vertical_spaces else 10.0

    unique_sorted_sizes = sorted(font_sizes.keys(), reverse=True)
    size_rank_map = {size: rank + 1 for rank, size in enumerate(unique_sorted_sizes)}

    return body_font_size, avg_vertical_space, size_rank_map

def extract_features_from_pdf(pdf_path):
    """
    Extracts a detailed feature vector for each TEXT LINE in a PDF.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening {pdf_path}: {e}")
        return []

    all_lines_features = []

    for page_num, page in enumerate(doc):
        body_font_size, avg_vertical_space, size_rank_map = get_page_stats(page)
        
        page_width = page.rect.width if page.rect.width > 0 else 1.0
        page_height = page.rect.height if page.rect.height > 0 else 1.0
        
        blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_SEARCH)["blocks"]
        
        all_lines = []
        for block in blocks:
            if block['type'] == 0 and 'lines' in block:
                for line_num, line in enumerate(block['lines']):
                    line['page_num'] = page_num
                    line['block_num'] = block['number']
                    line['line_num'] = line_num
                    all_lines.append(line)

        for i, line in enumerate(all_lines):
            if not line['spans']:
                continue

            first_span = line['spans'][0]
            line_text = clean_text(" ".join([span['text'] for span in line['spans']]))

            if not line_text:
                continue

            features = {}
            
            features['font_size'] = first_span['size']
            features['is_bold'] = 1 if (first_span['flags'] & 16) else 0
            features['is_italic'] = 1 if (first_span['flags'] & 2) else 0
            features['font_color'] = first_span['color']
            
            features['relative_font_size'] = features['font_size'] / body_font_size
            features['size_rank_on_page'] = size_rank_map.get(round(features['font_size']), 99)

            x0, y0, x1, y1 = line['bbox']
            features['is_centered_x'] = 1 - abs((x0 + x1) / 2 - page_width / 2) / (page_width / 2)
            features['x_pos_normalized'] = x0 / page_width
            features['y_pos_normalized'] = y0 / page_height
            features['block_width_normalized'] = (x1 - x0) / page_width
            features['block_height_normalized'] = (y1 - y0) / page_height
            
            features['word_count'] = len(line_text.split())
            features['char_count'] = len(line_text)
            features['starts_with_numbering'] = 1 if re.match(r'^\s*(\d+(\.\d+)*\.?|[A-Za-z]\.|[IVXLCDM]+\.)', line_text) else 0
            alpha_chars = sum(1 for char in line_text if char.isalpha())
            features['is_all_caps_ratio'] = sum(1 for char in line_text if char.isupper()) / (alpha_chars + 1e-6)
            features['is_title_case'] = 1 if is_title_case(line_text) else 0
            
            prev_line = all_lines[i - 1] if i > 0 else None
            next_line = all_lines[i + 1] if i < len(all_lines) - 1 else None
            
            features['vertical_space_above'] = line['bbox'][1] - prev_line['bbox'][3] if prev_line else 0
            features['vertical_space_below'] = next_line['bbox'][1] - line['bbox'][3] if next_line else 0
            features['is_new_block_group'] = 1 if features['vertical_space_above'] > (avg_vertical_space * 1.5) else 0

            if prev_line and prev_line['spans']:
                prev_span = prev_line['spans'][0]
                features['size_diff_with_prev'] = features['font_size'] - prev_span['size']
                prev_is_bold = 1 if (prev_span['flags'] & 16) else 0
                prev_is_italic = 1 if (prev_span['flags'] & 2) else 0
                features['is_font_style_change_prev'] = 1 if (features['is_bold'] != prev_is_bold or features['is_italic'] != prev_is_italic) else 0
            else:
                features['size_diff_with_prev'] = 0
                features['is_font_style_change_prev'] = 0

            if next_line and next_line['spans']:
                next_span = next_line['spans'][0]
                features['size_diff_with_next'] = features['font_size'] - next_span['size']
                next_is_bold = 1 if (next_span['flags'] & 16) else 0
                next_is_italic = 1 if (next_span['flags'] & 2) else 0
                features['is_font_style_change_next'] = 1 if (features['is_bold'] != next_is_bold or features['is_italic'] != next_is_italic) else 0
            else:
                features['size_diff_with_next'] = 0
                features['is_font_style_change_next'] = 0

            features['text'] = line_text
            features['page_num'] = line['page_num']
            features['block_num'] = line['block_num']
            features['line_num'] = line['line_num']

            all_lines_features.append(features)

    doc.close()
    return all_lines_features
