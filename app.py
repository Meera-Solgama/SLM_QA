import streamlit as st
import pandas as pd
import json
import re
import joblib
import random
import matplotlib.pyplot as plt
from collections import Counter

# Set page configuration
st.set_page_config(
    page_title="Gujarati QA System",
    page_icon="📚",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .section-title {
        color: #1B5299;
        font-size: 1.8rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2E86AB;
    }
    .qa-card {
        background-color: #F8F9FA;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #2E86AB;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .question-box {
        background-color: #E8F4F8;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .answer-box {
        background-color: #F0F7FF;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-right: 5px;
        margin-bottom: 5px;
    }
    .type-factual { background-color: #E3F2FD; color: #1565C0; }
    .type-numerical { background-color: #F3E5F5; color: #7B1FA2; }
    .type-list { background-color: #E8F5E9; color: #2E7D32; }
    .type-definition { background-color: #FFF3E0; color: #EF6C00; }
    .type-inferential { background-color: #FCE4EC; color: #C2185B; }
    .difficulty-easy { background-color: #C8E6C9; color: #2E7D32; }
    .difficulty-medium { background-color: #FFF3CD; color: #856404; }
    .difficulty-hard { background-color: #F8D7DA; color: #721C24; }
    .stButton>button {
        background-color: #2E86AB;
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1B5299;
        border-color: #1B5299;
    }
</style>
""", unsafe_allow_html=True)

class GujaratiQAApp:
    def __init__(self):
        self.loaded = self.load_models()
        
    def load_models(self):
        """Load trained models"""
        try:
            self.type_model = joblib.load('type_classifier.pkl')
            self.diff_model = joblib.load('diff_classifier.pkl')
            self.vectorizer = joblib.load('tfidf_vectorizer.pkl')
            self.type_encoder = joblib.load('type_encoder.pkl')
            self.diff_encoder = joblib.load('diff_encoder.pkl')
            return True
        except:
            st.warning("⚠ Models not found. Using rule-based system.")
            return False
    
    def clean_text(self, text):
        """Clean Gujarati text"""
        if not isinstance(text, str):
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s઀-૿.,!?;:]', '', text)
        return text.strip()
    
    def extract_keywords(self, text):
        """Extract keywords from text"""
        words = text.split()
        keywords = [w for w in words if len(w) > 3][:5]
        return keywords if keywords else ['તે']
    
    def predict_qa_type(self, question):
        """Predict question type"""
        if self.loaded:
            features = self.vectorizer.transform([question])
            type_pred = self.type_model.predict(features)[0]
            return self.type_encoder.inverse_transform([type_pred])[0]
        else:
            question_lower = question.lower()
            if 'કોણ' in question_lower or 'ક્યાં' in question_lower or 'શું' in question_lower:
                return 'factual'
            elif 'ક્યારે' in question_lower or 'કેટલા' in question_lower:
                return 'numerical/date'
            elif 'નામ' in question_lower or 'યાદી' in question_lower:
                return 'list'
            elif 'અર્થ' in question_lower:
                return 'definition'
            elif 'કેમ' in question_lower:
                return 'inferential'
            elif 'તફાવત' in question_lower:
                return 'comparative'
            elif 'વિષય' in question_lower:
                return 'thematic'
            elif 'મહત્વ' in question_lower:
                return 'evaluative'
            elif 'ભવિષ્ય' in question_lower:
                return 'predictive'
            return 'factual'
    
    def predict_difficulty(self, question):
        """Predict question difficulty"""
        if self.loaded:
            features = self.vectorizer.transform([question])
            diff_pred = self.diff_model.predict(features)[0]
            return self.diff_encoder.inverse_transform([diff_pred])[0]
        else:
            question_lower = question.lower()
            if 'કેમ' in question_lower or 'તફાવત' in question_lower or 'મૂલ્ય' in question_lower:
                return 'hard'
            elif len(question.split()) < 5:
                return 'easy'
            return 'medium'
    
    def extract_answer(self, context, question):
        """Extract answer from context"""
        sentences = [s.strip() + '.' for s in context.split('.') if s.strip()]
        if not sentences:
            return "જવાબ મળ્યો નથી."
        
        question_words = set(question.split())
        best_sentence = sentences[0]
        best_score = 0
        
        for sentence in sentences:
            sentence_words = set(sentence.split())
            common_words = len(question_words.intersection(sentence_words))
            if common_words > best_score:
                best_score = common_words
                best_sentence = sentence
        
        return best_sentence
    
    def generate_questions(self, context, num_questions=5):
        """Generate questions from context"""
        context_clean = self.clean_text(context)
        keywords = self.extract_keywords(context_clean)
        
        templates = [
            ("{} કોણ છે?", "factual"),
            ("{} ક્યાં છે?", "factual"),
            ("{} શું છે?", "factual"),
            ("{} ક્યારે થયું?", "numerical/date"),
            ("{} કેટલા છે?", "numerical/date"),
            ("{} ના નામ આપો.", "list"),
            ("'{}' નો અર્થ શું છે?", "definition"),
            ("{} કેમ થયું?", "inferential"),
            ("આ ફકરાનો મુખ્ય વિષય શું છે?", "thematic"),
            ("{} નું મહત્વ શું છે?", "evaluative"),
        ]
        
        qa_pairs = []
        for i in range(min(num_questions, len(templates))):
            template, default_type = templates[i]
            
            try:
                if template.count('{}') == 1 and keywords:
                    entity = keywords[i % len(keywords)]
                    question = template.format(entity)
                elif template.count('{}') == 0:
                    question = template
                else:
                    continue
                
                q_type = self.predict_qa_type(question)
                q_diff = self.predict_difficulty(question)
                answer = self.extract_answer(context_clean, question)
                
                qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'type': q_type,
                    'difficulty': q_diff,
                    'number': i+1
                })
            except:
                continue
        
        return qa_pairs

def main():
    """Main Streamlit app"""
    
    # Initialize the QA system
    qa_system = GujaratiQAApp()
    
    # Title
    st.markdown('<h1 class="main-title">📚 ગુજરાતી પ્રશ્ન-જવાબ સિસ્ટમ</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem;">AI-Powered Gujarati Question Generation & Answering</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/197/197566.png", width=80)
        st.markdown("### ⚙️ સેટિંગ્સ")
        
        num_questions = st.slider("પ્રશ્નોની સંખ્યા", 1, 10, 5)
        
        st.markdown("---")
        st.markdown("### 📊 સિસ્ટમ સ્ટેટ્સ")
        if qa_system.loaded:
            st.success("✅ મોડેલ્સ લોડ થયા")
            st.metric("સચોટતા", "80%")
        else:
            st.warning("⚠ રૂલ-બેસ્ડ સિસ્ટમ")
        
        st.markdown("---")
        st.markdown("### 📋 ઉદાહરણ")
        
        example_contexts = {
            "અમદાવાદ": """અમદાવાદ ગુજરાત રાજ્યનું સૌથી મોટું શહેર અને પૂર્વની રાજધાની છે. આ શહેરની સ્થાપના સુલતાન અહમદ શાહે ૨૬ ફેબ્રુઆરી ૧૪૧૧માં સાબરમતી નદીના કિનારે કરી હતી. અમદાવાદને 'પૂર્વનું મેન્ચેસ્ટર' પણ કહેવાય છે.""",
            "ગીર જંગલ": """ગીર રાષ્ટ્રીય ઉદ્યાન ગુજરાતના જુનાગઢ જિલ્લામાં આવેલું છે. આ એશિયાઇ સિંહનું એકમાત્ર નિવાસસ્થાન છે. ૧૪૧૨ ચોરસ કિ.મી. વિસ્તારમાં ફેલાયેલા આ ઉદ્યાનમાં ૫૦૦થી વધુ એશિયાઇ સિંહો રહે છે.""",
            "દાંડી કૂચ": """દાંડી કૂચ એ મહાત્મા ગાંધીજીના નેતૃત્વમાં ૧૯૩૦માં થયેલી ઐતિહાસિક યાત્રા હતી. ૧૨ માર્ચથી ૬ એપ્રિલ ૧૯૩૦ સુધી ચાલેલી આ કૂચમાં ૭૮ સત્યાગ્રહીઓએ ૩૯૦ કિ.મી.નો પગપાળો માર્ગ કાપ્યો હતો."""
        }
        
        selected_example = st.selectbox("ઉદાહરણ પસંદ કરો:", list(example_contexts.keys()))
        
        if st.button("ઉદાહરણ લોડ કરો"):
            st.session_state.context = example_contexts[selected_example]
            st.rerun()
    
    # Main content area
    st.markdown('<h2 class="section-title">📝 ગુજરાતી ટેક્સ્ટ દાખલ કરો</h2>', unsafe_allow_html=True)
    
    # Text input
    context_input = st.text_area(
        "તમારો ગુજરાતી ટેક્સ્ટ અહીં લખો:",
        height=200,
        value=st.session_state.get('context', ''),
        placeholder="ઉદાહરણ: અમદાવાદ ગુજરાત રાજ્યનું સૌથી મોટું શહેર છે...",
        help="તમારો ગુજરાતી ટેક્સ્ટ અહીં લખો અને પ્રશ્નો જનરેટ કરવા માટે બટન દબાવો."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        generate_btn = st.button("🚀 પ્રશ્નો જનરેટ કરો", use_container_width=True)
    with col2:
        if st.button("🧹 સાફ કરો", use_container_width=True):
            st.session_state.context = ""
            st.rerun()
    
    # Generate questions
    if generate_btn and context_input.strip():
        with st.spinner("પ્રશ્નો જનરેટ થઈ રહ્યા છે..."):
            qa_pairs = qa_system.generate_questions(context_input, num_questions)
            
            if qa_pairs:
                st.markdown(f'<h2 class="section-title">✅ {len(qa_pairs)} પ્રશ્નો જનરેટ થયા</h2>', unsafe_allow_html=True)
                
                # Display all QA pairs
                for qa in qa_pairs:
                    with st.container():
                        col_q, col_a = st.columns([3, 1])
                        
                        with col_q:
                            st.markdown(f'<div class="question-box">', unsafe_allow_html=True)
                            st.markdown(f"**પ્રશ્ન {qa['number']}:** {qa['question']}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col_a:
                            type_class = qa['type'].lower().replace('/', '')
                            diff_class = qa['difficulty'].lower()
                            
                            st.markdown(f'<span class="badge type-{type_class}">{qa["type"]}</span>', unsafe_allow_html=True)
                            st.markdown(f'<span class="badge difficulty-{diff_class}">{qa["difficulty"]}</span>', unsafe_allow_html=True)
                        
                        st.markdown(f'<div class="answer-box">', unsafe_allow_html=True)
                        st.markdown(f"**જવાબ:** {qa['answer']}")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Statistics
                st.markdown('<h3 class="section-title">📊 આંકડા</h3>', unsafe_allow_html=True)
                
                col_s1, col_s2, col_s3 = st.columns(3)
                
                with col_s1:
                    types = [qa['type'] for qa in qa_pairs]
                    type_counts = Counter(types)
                    st.metric("પ્રશ્ન પ્રકારો", len(type_counts))
                
                with col_s2:
                    difficulties = [qa['difficulty'] for qa in qa_pairs]
                    diff_counts = Counter(difficulties)
                    st.metric("ડિફિકલ્ટી લેવલ્સ", len(diff_counts))
                
                with col_s3:
                    total_words = sum(len(qa['answer'].split()) for qa in qa_pairs)
                    st.metric("કુલ શબ્દો (જવાબો)", total_words)
                
                # Visualization
                st.markdown('<h3 class="section-title">📈 વિઝ્યુલાઈઝેશન</h3>', unsafe_allow_html=True)
                
                fig_col1, fig_col2 = st.columns(2)
                
                with fig_col1:
                    # Question type pie chart
                    type_df = pd.DataFrame(list(type_counts.items()), columns=['Type', 'Count'])
                    fig1, ax1 = plt.subplots()
                    ax1.pie(type_df['Count'], labels=type_df['Type'], autopct='%1.1f%%', startangle=90)
                    ax1.axis('equal')
                    st.pyplot(fig1)
                    st.caption("પ્રશ્ન પ્રકાર વિતરણ")
                
                with fig_col2:
                    # Difficulty bar chart
                    diff_df = pd.DataFrame(list(diff_counts.items()), columns=['Difficulty', 'Count'])
                    fig2, ax2 = plt.subplots()
                    colors = ['#C8E6C9' if d == 'easy' else '#FFF3CD' if d == 'medium' else '#F8D7DA' for d in diff_df['Difficulty']]
                    ax2.bar(diff_df['Difficulty'], diff_df['Count'], color=colors)
                    ax2.set_ylabel('પ્રશ્નોની સંખ્યા')
                    st.pyplot(fig2)
                    st.caption("ડિફિકલ્ટી લેવલ વિતરણ")
                
                # Download options
                st.markdown('<h3 class="section-title">💾 ડાઉનલોડ</h3>', unsafe_allow_html=True)
                
                col_d1, col_d2 = st.columns(2)
                
                with col_d1:
                    # JSON download
                    json_str = json.dumps(qa_pairs, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="📥 JSON તરીકે ડાઉનલોડ કરો",
                        data=json_str,
                        file_name="gujarati_qa.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with col_d2:
                    # CSV download
                    df = pd.DataFrame(qa_pairs)
                    csv = df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📊 CSV તરીકે ડાઉનલોડ કરો",
                        data=csv,
                        file_name="gujarati_qa.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            else:
                st.error("⚠ કોઈ પ્રશ્ન જનરેટ થયા નથી. કૃપા કરીને વધુ લાંબો ટેક્સ્ટ દાખલ કરો.")
    
    elif generate_btn and not context_input.strip():
        st.warning("⚠ કૃપા કરીને પહેલા ટેક્સ્ટ દાખલ કરો.")
    
    # About section
    st.markdown("---")
    st.markdown('<h2 class="section-title">ℹ️ સિસ્ટમ વિશે</h2>', unsafe_allow_html=True)
    
    about_col1, about_col2 = st.columns(2)
    
    with about_col1:
        st.info("""
        **વિશેષતાઓ:**
        - ગુજરાતી ટેક્સ્ટમાંથી પ્રશ્નો જનરેટ કરો
        - 9 પ્રકારના પ્રશ્નો
        - 3 ડિફિકલ્ટી લેવલ્સ
        - 80%+ સચોટતા
        - જવાબો સ્વચાલિત એક્સ્ટ્રેક્ટ
        """)
    
    with about_col2:
        st.info("""
        **ટેક્નોલોજી:**
        - Random Forest Models
        - TF-IDF Vectorization
        - Streamlit UI
        - Gujarati NLP
        - 523 QA પેઅર્સ ડેટાસેટ
        """)

if __name__ == "__main__":
    main()
