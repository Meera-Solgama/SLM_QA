import streamlit as st
import pandas as pd
import json
import re
import joblib
import random
from collections import Counter
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="Gujarati QA", page_icon="📚", layout="wide")

# Title
st.title("📚 ગુજરાતી પ્રશ્ન-જવાબ સિસ્ટમ")
st.markdown("---")

# Try to load models
@st.cache_resource
def load_models():
    try:
        type_model = joblib.load('type_classifier.pkl')
        diff_model = joblib.load('diff_classifier.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        type_encoder = joblib.load('type_encoder.pkl')
        diff_encoder = joblib.load('diff_encoder.pkl')
        return type_model, diff_model, vectorizer, type_encoder, diff_encoder, True
    except:
        return None, None, None, None, None, False

# Load models
type_model, diff_model, vectorizer, type_encoder, diff_encoder, models_loaded = load_models()

if models_loaded:
    st.success("✅ મોડેલ્સ લોડ થયા!")
else:
    st.info("ℹ️ રૂલ-બેસ્ડ સિસ્ટમ વાપરી રહ્યા છીએ")

# Text input
context = st.text_area("તમારો ગુજરાતી ટેક્સ્ટ લખો:", height=200, 
                      placeholder="ઉદાહરણ: અમદાવાદ ગુજરાત રાજ્યનું સૌથી મોટું શહેર છે...")

# Number of questions
num_q = st.slider("પ્રશ્નોની સંખ્યા", 1, 10, 3)

# Generate button
if st.button("🚀 પ્રશ્નો જનરેટ કરો") and context:
    
    with st.spinner("પ્રશ્નો જનરેટ થઈ રહ્યા છે..."):
        
        # Clean text
        context_clean = re.sub(r'\s+', ' ', context)
        context_clean = re.sub(r'[^\w\s઀-૿.,!?;:]', '', context_clean)
        
        # Simple keyword extraction
        words = context_clean.split()
        keywords = [w for w in words if len(w) > 3][:5]
        
        # Question templates
        templates = [
            ("{} કોણ છે?", "factual"),
            ("{} ક્યાં છે?", "factual"), 
            ("{} શું છે?", "definition"),
            ("{} ક્યારે થયું?", "numerical"),
            ("{} કેટલા છે?", "numerical"),
        ]
        
        qa_pairs = []
        
        for i in range(min(num_q, len(templates))):
            if keywords:
                entity = keywords[i % len(keywords)]
                question = templates[i][0].format(entity)
                q_type = templates[i][1]
                
                # Simple answer extraction
                sentences = [s.strip() + '.' for s in context_clean.split('.') if s.strip()]
                answer = sentences[0] if sentences else "જવાબ મળ્યો નથી."
                
                # Predict difficulty (simple rule)
                if len(question.split()) < 4:
                    difficulty = "easy"
                elif "ક્યારે" in question or "કેટલા" in question:
                    difficulty = "medium"
                else:
                    difficulty = "hard"
                
                qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "type": q_type,
                    "difficulty": difficulty
                })
        
        # Show results
        st.success(f"✅ {len(qa_pairs)} પ્રશ્નો જનરેટ થયા!")
        
        for i, qa in enumerate(qa_pairs):
            st.markdown(f"**પ્રશ્ન {i+1}:** {qa['question']}")
            st.markdown(f"**જવાબ:** {qa['answer']}")
            st.markdown(f"**પ્રકાર:** {qa['type']} | **ડિફિકલ્ટી:** {qa['difficulty']}")
            st.markdown("---")
        
        # Download option
        json_data = json.dumps(qa_pairs, ensure_ascii=False, indent=2)
        st.download_button("📥 ડાઉનલોડ કરો", json_data, "gujarati_qa.json", "application/json")

else:
    if not context:
        st.warning("⚠ કૃપા કરીને ટેક્સ્ટ લખો")

# Footer
st.markdown("---")
st.markdown("© 2024 Gujarati QA System | Made with ❤️")
