import os
import joblib
import pandas as pd
from .feature_extractor import extract_features_from_pdf

HEADING_HIERARCHY = {
    "Title": 0, "H1": 1, "H2": 2, "H3": 3, "H4": 4, "H5": 5, "H6": 6
}

def group_text_into_sections(labeled_lines, pdf_filename):
    sections = []
    current_section = None
    active_heading_stack = []

    for item in labeled_lines:
        label = item.get('label')
        text = item.get('text', '').strip()
        page_num = item.get('page')

        if not text:
            continue

        is_heading = label in HEADING_HIERARCHY

        if is_heading:
            if current_section:
                current_section['content'] = ' '.join(current_section['content'].split())
                sections.append(current_section)

            heading_level = HEADING_HIERARCHY[label]
            
            while active_heading_stack and active_heading_stack[-1]['level'] >= heading_level:
                active_heading_stack.pop()
            
            current_section = {
                "document_name": pdf_filename,
                "page_number": page_num,
                "section_title": text,
                "content": "",
                "hierarchy_level": heading_level,
                "full_path": [h['title'] for h in active_heading_stack] + [text]
            }
            active_heading_stack.append({'title': text, 'level': heading_level})

        elif label == 'Body' and current_section:
            current_section['content'] += f" {text}"
    
    if current_section:
        current_section['content'] = ' '.join(current_section['content'].split())
        sections.append(current_section)

    return sections

def add_full_content_to_sections(sections):
    for section in sections:
        path_str = " > ".join(section["full_path"])
        section["full_content"] = f"{path_str}: {section['content']}"
    return sections

def parse_document_to_sections(pdf_path, model_path, encoder_path):
    pdf_filename = os.path.basename(pdf_path)
    print(f"Processing '{pdf_filename}'...")

    try:
        model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
    except Exception as e:
        print(f"Error loading model/encoder: {e}")
        return None

    features_list = extract_features_from_pdf(pdf_path)
    if not features_list:
        print("Could not extract any features from the PDF.")
        return None
    
    df = pd.DataFrame(features_list)
    
    try:
        model_features = model.feature_names_in_
    except AttributeError:
        model_features = [col for col in df.columns if col not in ['text', 'page_num', 'block_num', 'line_num']]

    for col in model_features:
        if col not in df.columns:
            df[col] = 0
            
    X_predict = df[model_features]

    predictions_encoded = model.predict(X_predict)
    predictions_labels = label_encoder.inverse_transform(predictions_encoded)
    df['predicted_label'] = predictions_labels

    labeled_lines = []
    for _, row in df.iterrows():
        if row['predicted_label'] != 'Other':
            labeled_lines.append({
                "label": row['predicted_label'],
                "text": row['text'],
                "page": int(row['page_num'])
            })
    
    structured_sections = group_text_into_sections(labeled_lines, pdf_filename)
    final_sections = add_full_content_to_sections(structured_sections)
    
    print(f"Successfully parsed into {len(final_sections)} sections.")
    
    return final_sections
