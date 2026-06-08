import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configurazione pagina
st.set_page_config(page_title="AI & Student Performance Analytics", layout="wide")

# 1. FUNZIONI PER IL CARICAMENTO DEI MODELLI
@st.cache_resource
def load_models():
    # Sostituisci i nomi dei file con i tuoi modelli reali salvati
    try:
        model_gpa = joblib.load("modello_gpa.pkl")
        model_burnout = joblib.load("modello_burnout.pkl")
        return model_gpa, model_burnout
    except FileNotFoundError:
        # Modelli fittizi di fallback se non trova i file (giusto per non far crashare l'app al primo avvio)
        return None, None

model_gpa, model_burnout = load_models()

# 2. INTERFACCIA GRAFICA (SIDEBAR & TITOLO)
st.title("🎓 Impatto della GenAI sulle Performance degli Studenti")
st.markdown("Inserisci i dati dello studente per calcolare la predizione del **GPA** o del **Rischio di Burnout**.")

st.sidebar.header("Impostazioni Predizione")
scelta_predizione = st.sidebar.radio(
    "Cosa vuoi predire?",
    ["Post_Semester_GPA (Regressione)", "Burnout_Risk_Level (Classificazione)"]
)



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

# Feature extra se necessaria per il modello burnout
with col1:
    if "Burnout" in scelta_predizione:
        retention = st.slider("Skill Retention Score", 0.0, 100.0, 70.0)
    else:
        retention = 70.0 # Valore di default se non usato


# 4. PREPARAZIONE DEI DATI PER SCIKIT-LEARN
# Nota: Se nel tuo modello hai usato la codifica One-Hot (get_dummies) o LabelEncoder per il testo,
# devi replicare lo stesso identico formato qui prima di passarlo al `.predict()`.

dati_utente = pd.DataFrame([{
    'Major_Category': major,
    'Year_of_Study': year,
    'Pre_Semester_GPA': pre_gpa,
    'Weekly_GenAI_Hours': weekly_hours,
    'Primary_Use_Case': use_case,
    'Prompt_Engineering_Skill': prompt_skill,
    'Tool_Diversity': tool_div,
    'Paid_Subscription': paid_sub,
    'Traditional_Study_Hours': study_hours,
    'Perceived_AI_Dependency': ai_dep,
    'Institutional_Policy': policy,
    'Anxiety_Level_During_Exams': anxiety,
    'Skill_Retention_Score': retention
}])

# Mostra i dati inseriti in formato tabella (opzionale, utile per debug)
if st.checkbox("Mostra DataFrame di Input inviato al modello"):
    st.dataframe(dati_utente)



# 5. PULSANTE DI CALCOLO E OUTPUT
st.write("") # Spazio vuoto
if st.button("🚀 Calcola Predizione", type="primary"):
    
    # ESEMPIO DI PRE-PROCESSING (Adatta in base a come hai allenato il modello)
    # Se il tuo modello accetta solo numeri, dovrai mappare le stringhe in numeri (es. map o OneHotEncoder)
    # Per ora simuliamo il passaggio diretto o mostriamo un avviso se il modello manca:
    
    if "GPA" in scelta_predizione:
        if model_gpa is not None:
            # Sostituisci questa riga con il pre-processing corretto se necessario
            pred = model_gpa.predict(dati_utente) 
            st.metric(label="🎯 Post_Semester_GPA Predetto", value=f"{pred[0]:.3f}")
        else:
            st.warning("⚠️ Modello 'modello_gpa.pkl' non trovato. Ecco una predizione simulata:")
            simulated_gpa = max(0.0, min(4.0, pre_gpa - (weekly_hours * 0.01) + (study_hours * 0.02)))
            st.metric(label="🎯 Post_Semester_GPA (Simulato)", value=f"{simulated_gpa:.3f}")
            
    else: # Predizione Burnout
        if model_burnout is not None:
            pred = model_burnout.predict(dati_utente)
            st.subheader(f"🔥 Livello Rischio Burnout: {pred[0]}")
        else:
            st.warning("⚠️ Modello 'modello_burnout.pkl' non trovato. Ecco una predizione simulata:")
            rischio = "High" if (weekly_hours > 20 and anxiety > 7) else "Medium" if anxiety > 4 else "Low"
            
            if rischio == "High":
                st.error(f"🔥 Livello Rischio Burnout (Simulato): {rischio}")
            elif rischio == "Medium":
                st.warning(f"⚠️ Livello Rischio Burnout (Simulato): {rischio}")
            else:
                st.success(f"✅ Livello Rischio Burnout (Simulato): {rischio}")