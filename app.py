import streamlit as st
from transformers import pipeline
import torch
import json, os  # <-- add here

@st.cache_resource
def load_generator():
    return pipeline('text-generation', model='D:/fake_news_project/gpt2_out',
                    tokenizer='D:/fake_news_project/gpt2_out',
                    device=0 if torch.cuda.is_available() else -1)

@st.cache_resource
def load_classifier():
    return pipeline('text-classification', model='D:/fake_news_project/detector_out',
                    tokenizer='D:/fake_news_project/detector_out',
                    return_all_scores=True,
                    device=0 if torch.cuda.is_available() else -1)

# helper: read id2label from saved config if present, else fallback
def read_id2label(model_dir='D:/fake_news_project/detector_out'):
    cfg_path = os.path.join(model_dir, 'config.json')
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            if 'id2label' in cfg:
                return {int(k): v for k, v in cfg['id2label'].items()}
        except Exception:
            pass
    return {0: 'REAL', 1: 'FAKE'}

st.title('Fake News Generator & Detector')

option = st.selectbox('Choose action', ['Generate', 'Detect'])

if option == 'Generate':
    prompt = st.text_input('Enter prompt', 'Breaking:')
    if st.button('Generate'):
        gen = load_generator()
        outputs = gen(prompt, max_length=30, num_return_sequences=3, do_sample=True)
        for o in outputs:
            st.write(o['generated_text'])
else:
    text = st.text_area('Enter text', height=150)
    if st.button('Detect'):
        if not text or not text.strip():
            st.warning('Please enter a headline or short text to classify.')
        else:
            with st.spinner('Loading classifier (if not cached)...'):
                clf = load_classifier()
            with st.spinner('Classifying text...'):
                raw_scores = clf(text)[0]

            id2label_map = read_id2label('D:/fake_news_project/detector_out')
            results = []
            for item in raw_scores:
                lab = item['label']
                if isinstance(lab, str) and lab.startswith('LABEL_'):
                    try:
                        idx = int(lab.split('_')[1])
                    except:
                        idx = None
                else:
                    try:
                        idx = int(lab)
                    except:
                        idx = None
                human = id2label_map.get(idx, lab)
                results.append((human, item['score']))

            st.subheader('Probabilities')
            for human, score in results:
                st.write(f"{human}: {score:.3f}")

            best_item = max(raw_scores, key=lambda x: x['score'])
            if best_item['label'].startswith('LABEL_'):
                best_idx = int(best_item['label'].split('_')[1])
            else:
                best_idx = int(best_item['label'])
            best_label = id2label_map.get(best_idx, f'LABEL_{best_idx}')
            st.success(f"Prediction: {best_label} (confidence {best_item['score']:.2f})")
