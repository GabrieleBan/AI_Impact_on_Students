import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configurazione pagina
st.set_page_config(page_title="AI & Student Performance Analytics", layout="wide")

# 1. FUNZIONI PER IL CARICAMENTO DEI MODELLI
@st.cache_resource
def load_models(path_model):
    try:
        model = joblib.load(path_model)
        return model
    except FileNotFoundError:
        return None

# Caricamento dei file reali
model_gpa = load_models("gpa_model.pkl")
model_burnout = load_models("burnout_model.pkl")

# 2. INTERFACCIA GRAFICA (TITOLO)
st.title("🎓 Impatto della GenAI sulle Performance degli Studenti")
st.markdown("Inserisci i dati dello studente per calcolare simultaneamente la predizione del **GPA** e del **Rischio di Burnout**.")

# 3. INPUT DELL'UTENTE (Organizzato in colonne)
st.subheader("📝 Dati dello Studente")

col1, col2, col3 = st.columns(3)

with col1:
    major = col1.selectbox("Facoltà (Major_Category)", ["Humanities", "Medical", "Business", "STEM", "Other"])
    year = col1.selectbox("Anno di Studio", ["Freshman", "Sophomore", "Junior", "Senior"])
    pre_gpa = col1.number_input("Pre_Semester_GPA", min_value=0.0, max_value=4.0, value=3.0, step=0.1)
    weekly_hours = col1.number_input("Ore settimanali uso GenAI", min_value=0.0, max_value=50.0, value=10.0)

with col2:
    use_case = col2.selectbox("Uso Principale AI", ["Copywriting/Drafting", "Ideation", "Summarizing_Reading", "Coding", "Research"])
    prompt_skill = col2.selectbox("Skill Prompt Engineering", ["Beginner", "Intermediate", "Advanced"])
    tool_div = col2.slider("Diversità Strumenti Usati (Tool Diversity)", 1, 10, 3)
    paid_sub = col2.checkbox("Abbonamento GenAI a pagamento?")

with col3:
    study_hours = col3.number_input("Ore Studio Tradizionale/Settimana", min_value=0.0, max_value=100.0, value=15.0)
    ai_dep = col3.slider("Dipendenza percepita da AI (1-5)", 1, 5, 3)
    policy = col3.selectbox("Policy Istituzionale AI", ["Allowed_With_Citation", "Strict_Ban", "Open_Use"])
    anxiety = col3.slider("Livello Ansia Esami (1-10)", 1, 10, 5)

with col1:
    retention = st.slider("Skill Retention Score", 0.0, 100.0, 70.0)


# 4. PREPARAZIONE DEI DATI ADATTATA AI DUE DIFFERENTI MODELLI

# --- STRUTTURA A: DATI PER IL MODELLO GPA (22 colonne specifiche) ---
features_gpa = {
    'Year_of_Study': year, 'Pre_Semester_GPA': pre_gpa, 'Weekly_GenAI_Hours': weekly_hours,
    'Prompt_Engineering_Skill': prompt_skill, 'Tool_Diversity': tool_div, 'Paid_Subscription': paid_sub,
    'Traditional_Study_Hours': study_hours, 'Perceived_AI_Dependency': ai_dep, 'Anxiety_Level_During_Exams': anxiety,
    'is_Copywriting/Drafting': False, 'is_Ideation': False, 'is_Summarizing_Reading': False,
    'is_Debugging/Troubleshooting': False, 'is_Direct_Answer_Generation': False,
    'is_Allowed_With_Citation': False, 'is_Strict_Ban': False, 'is_Actively_Encouraged': False,
    'major_is_Humanities': False, 'major_is_Medical': False, 'major_is_Business': False,
    'major_is_STEM': False, 'major_is_Arts': False
}

if use_case == "Copywriting/Drafting": features_gpa['is_Copywriting/Drafting'] = True
elif use_case == "Ideation": features_gpa['is_Ideation'] = True
elif use_case == "Summarizing_Reading": features_gpa['is_Summarizing_Reading'] = True
elif use_case == "Coding": features_gpa['is_Debugging/Troubleshooting'] = True
elif use_case == "Research": features_gpa['is_Direct_Answer_Generation'] = True

if policy == "Allowed_With_Citation": features_gpa['is_Allowed_With_Citation'] = True
elif policy == "Strict_Ban": features_gpa['is_Strict_Ban'] = True
elif policy == "Open_Use": features_gpa['is_Actively_Encouraged'] = True

if major == "Humanities": features_gpa['major_is_Humanities'] = True
elif major == "Medical": features_gpa['major_is_Medical'] = True
elif major == "Business": features_gpa['major_is_Business'] = True
elif major == "STEM": features_gpa['major_is_STEM'] = True
elif major == "Other": features_gpa['major_is_Arts'] = True

ordine_gpa = [
    'Year_of_Study', 'Pre_Semester_GPA', 'Weekly_GenAI_Hours', 'Prompt_Engineering_Skill', 
    'Tool_Diversity', 'Paid_Subscription', 'Traditional_Study_Hours', 'Perceived_AI_Dependency', 
    'Anxiety_Level_During_Exams', 'is_Copywriting/Drafting', 'is_Ideation', 'is_Summarizing_Reading', 
    'is_Debugging/Troubleshooting', 'is_Direct_Answer_Generation', 'is_Allowed_With_Citation', 
    'is_Strict_Ban', 'is_Actively_Encouraged', 'major_is_Humanities', 'major_is_Medical', 
    'major_is_Business', 'major_is_STEM', 'major_is_Arts'
]
dati_gpa = pd.DataFrame([features_gpa])[ordine_gpa]
dati_gpa['Year_of_Study'] = dati_gpa['Year_of_Study'].astype('category')
dati_gpa['Prompt_Engineering_Skill'] = dati_gpa['Prompt_Engineering_Skill'].astype('category')


# --- STRUTTURA B: DATI PER IL MODELLO BURNOUT (22 colonne estratte dal log di errore) ---
# Eseguiamo prima la predizione del GPA per poter passare il valore 'Post_Semester_GPA' richiesto dal modello Burnout
simulated_or_predicted_gpa = pre_gpa # Valore di fallback

if model_gpa is not None:
    try:
        simulated_or_predicted_gpa = model_gpa.predict(dati_gpa)[0]
    except Exception:
        pass

features_burnout = {
    # Variabili numeriche/scalari standard
    'Year_of_Study': ["Freshman", "Sophomore", "Junior", "Senior"].index(year), # Trasformato in numerico per stabilità
    'Pre_Semester_GPA': float(pre_gpa),
    'Weekly_GenAI_Hours': float(weekly_hours),
    'Tool_Diversity': int(tool_div),
    'Paid_Subscription': int(paid_sub),
    'Traditional_Study_Hours': float(study_hours),
    'Perceived_AI_Dependency': int(ai_dep),
    'Anxiety_Level_During_Exams': int(anxiety),
    'Post_Semester_GPA': float(simulated_or_predicted_gpa),
    'Skill_Retention_Score': float(retention),

    # One-Hot Encoding puro richiesto dall'errore
    'Major_Category_Business': 1 if major == "Business" else 0,
    'Major_Category_Humanities': 1 if major == "Humanities" else 0,
    'Major_Category_Medical': 1 if major == "Medical" else 0,
    'Major_Category_STEM': 1 if major == "STEM" else 0,
    
    'Primary_Use_Case_Debugging/Troubleshooting': 1 if use_case == "Coding" else 0,
    'Primary_Use_Case_Direct_Answer_Generation': 1 if use_case == "Research" else 0,
    'Primary_Use_Case_Ideation': 1 if use_case == "Ideation" else 0,
    'Primary_Use_Case_Summarizing_Reading': 1 if use_case == "Summarizing_Reading" else 0,
    
    'Prompt_Engineering_Skill_Beginner': 1 if prompt_skill == "Beginner" else 0,
    'Prompt_Engineering_Skill_Intermediate': 1 if prompt_skill == "Intermediate" else 0,
    
    'Institutional_Policy_Allowed_With_Citation': 1 if policy == "Allowed_With_Citation" else 0,
    'Institutional_Policy_Strict_Ban': 1 if policy == "Strict_Ban" else 0
}

# Ordine esatto richiesto dal tracciato di addestramento del modello Burnout
ordine_burnout = [
    'Major_Category_Business', 'Major_Category_Humanities', 'Major_Category_Medical', 'Major_Category_STEM',
    'Primary_Use_Case_Debugging/Troubleshooting', 'Primary_Use_Case_Direct_Answer_Generation', 
    'Primary_Use_Case_Ideation', 'Primary_Use_Case_Summarizing_Reading', 
    'Prompt_Engineering_Skill_Beginner', 'Prompt_Engineering_Skill_Intermediate', 
    'Institutional_Policy_Allowed_With_Citation', 'Institutional_Policy_Strict_Ban', 
    'Year_of_Study', 'Pre_Semester_GPA', 'Weekly_GenAI_Hours', 'Tool_Diversity', 
    'Paid_Subscription', 'Traditional_Study_Hours', 'Perceived_AI_Dependency', 
    'Anxiety_Level_During_Exams', 'Post_Semester_GPA', 'Skill_Retention_Score'
]

dati_burnout = pd.DataFrame([features_burnout])[ordine_burnout]


# Checkbox di Debug per visualizzare l'allineamento perfetto
if st.checkbox("Mostra DataFrame di Input inviati ai modelli"):
    st.markdown("**Tracciato Modello GPA:**")
    st.dataframe(dati_gpa)
    st.markdown("**Tracciato Modello Burnout (Allineato all'errore):**")
    st.dataframe(dati_burnout)


# 5. PULSANTE DI CALCOLO UNICO E OUTPUT
st.write("") 
if st.button("🚀 Calcola Predizioni", type="primary", use_container_width=True):
    
    st.markdown("### 📊 Risultati dell'Analisi")
    res_col1, res_col2 = st.columns(2)
    
    # --- COLONNA 1: PREDIZIONE GPA ---
    with res_col1:
        st.markdown("#### 🎯 Performance Accademica")
        if model_gpa is not None:
            pred_gpa = model_gpa.predict(dati_gpa)[0] 
            st.metric(label="Post_Semester_GPA Predetto", value=f"{pred_gpa:.3f}")
        else:
            st.warning("⚠️ File 'gpa_model.pkl' non trovato. Mostro simulazione:")
            simulated_gpa = max(0.0, min(4.0, pre_gpa - (weekly_hours * 0.01) + (study_hours * 0.02)))
            st.metric(label="Post_Semester_GPA (Simulato)", value=f"{simulated_gpa:.3f}")

    # --- COLONNA 2: PREDIZIONE BURNOUT ---
    with res_col2:
        st.markdown("#### 🧠 Benessere dello Studente")
        if model_burnout is not None:
            # Se la predizione del GPA reale è avvenuta con successo, aggiorna il record dinamico per il Burnout
            if model_gpa is not None:
                dati_burnout['Post_Semester_GPA'] = float(pred_gpa)
                
            pred_burnout = model_burnout.predict(dati_burnout)[0]
            
            if str(pred_burnout).lower() == "high":
                st.error(f"🔥 Livello Rischio Burnout: {pred_burnout}")
            elif str(pred_burnout).lower() == "medium":
                st.warning(f"⚠️ Livello Rischio Burnout: {pred_burnout}")
            else:
                st.success(f"✅ Livello Rischio Burnout: {pred_burnout}")
        else:
            st.warning("⚠️ File 'burnout_model.pkl' non trovato. Mostro simulazione:")
            rischio = "High" if (weekly_hours > 20 and anxiety > 7) else "Medium" if anxiety > 4 else "Low"
            
            if rischio == "High":
                st.error(f"🔥 Livello Rischio Burnout (Simulato): {rischio}")
            elif rischio == "Medium":
                st.warning(f"⚠️ Livello Rischio Burnout (Simulato): {rischio}")
            else:
                st.success(f"✅ Livello Rischio Burnout (Simulato): {rischio}")