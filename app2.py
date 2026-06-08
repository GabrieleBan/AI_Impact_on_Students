import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm # Importato ma non usato direttamente se i pkl sono XGBoost

# Configurazione pagina
st.set_page_config(page_title="AI & Student Performance Analytics", layout="wide")

# 1. FUNZIONI PER IL CARICAMENTO DEI MODELLI
@st.cache_resource
def load_models(path_model):
    try:
        model = joblib.load(path_model)
        return model
    except FileNotFoundError:
        # Modelli fittizi di fallback se non trova i file
        return None

# Caricamento dei file reali aggiornati
model_gpa = load_models("gpa_model.pkl")
model_burnout = load_models("burnout_model.pkl")

# Configurazione parametri LightGBM interni al VotingClassifier per ignorare shape mismatch minori
if model_burnout is not None:
    try:
        if hasattr(model_burnout, "estimators_"):
            for est in model_burnout.estimators_:
                if 'LightGBM' in str(type(est)) or hasattr(est, "set_params"):
                    est.set_params(predict_disable_shape_check=True)
    except Exception:
        pass

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

# Skill retention posizionata sotto la colonna 1
with col1:
    retention = st.slider("Skill Retention Score", 0.0, 100.0, 70.0)


# 4. PREPARAZIONE DEI DATI - TRASFORMAZIONE ONE-HOT ENCODING UNIFICATA (22 colonne ordinate)
features_comuni = {
    'Year_of_Study': year,
    'Pre_Semester_GPA': pre_gpa,
    'Weekly_GenAI_Hours': weekly_hours,
    'Prompt_Engineering_Skill': prompt_skill,
    'Tool_Diversity': tool_div,
    'Paid_Subscription': paid_sub,
    'Traditional_Study_Hours': study_hours,
    'Perceived_AI_Dependency': ai_dep,
    'Anxiety_Level_During_Exams': anxiety,
    
    # Inizializziamo a False i booleani del Caso d'Uso
    'is_Copywriting/Drafting': False,
    'is_Ideation': False,
    'is_Summarizing_Reading': False,
    'is_Debugging/Troubleshooting': False,
    'is_Direct_Answer_Generation': False,
    
    # Inizializziamo a False le Policy istituzionali
    'is_Allowed_With_Citation': False,
    'is_Strict_Ban': False,
    'is_Actively_Encouraged': False,
    
    # Inizializziamo a False i Major universitari
    'major_is_Humanities': False,
    'major_is_Medical': False,
    'major_is_Business': False,
    'major_is_STEM': False,
    'major_is_Arts': False
}

# Attiviamo il flag corretto in base all'input selezionato dall'utente
if use_case == "Copywriting/Drafting": features_comuni['is_Copywriting/Drafting'] = True
elif use_case == "Ideation": features_comuni['is_Ideation'] = True
elif use_case == "Summarizing_Reading": features_comuni['is_Summarizing_Reading'] = True
elif use_case == "Coding": features_comuni['is_Debugging/Troubleshooting'] = True
elif use_case == "Research": features_comuni['is_Direct_Answer_Generation'] = True

if policy == "Allowed_With_Citation": features_comuni['is_Allowed_With_Citation'] = True
elif policy == "Strict_Ban": features_comuni['is_Strict_Ban'] = True
elif policy == "Open_Use": features_comuni['is_Actively_Encouraged'] = True

if major == "Humanities": features_comuni['major_is_Humanities'] = True
elif major == "Medical": features_comuni['major_is_Medical'] = True
elif major == "Business": features_comuni['major_is_Business'] = True
elif major == "STEM": features_comuni['major_is_STEM'] = True
elif major == "Other": features_comuni['major_is_Arts'] = True

# Ordine tassativo richiesto da XGBoost e LightGBM (22 colonne esatte)
ordine_colonne_finale = [
    'Year_of_Study', 'Pre_Semester_GPA', 'Weekly_GenAI_Hours', 'Prompt_Engineering_Skill', 
    'Tool_Diversity', 'Paid_Subscription', 'Traditional_Study_Hours', 'Perceived_AI_Dependency', 
    'Anxiety_Level_During_Exams', 'is_Copywriting/Drafting', 'is_Ideation', 'is_Summarizing_Reading', 
    'is_Debugging/Troubleshooting', 'is_Direct_Answer_Generation', 'is_Allowed_With_Citation', 
    'is_Strict_Ban', 'is_Actively_Encouraged', 'major_is_Humanities', 'major_is_Medical', 
    'major_is_Business', 'major_is_STEM', 'major_is_Arts'
]

# Creiamo l'unico vero DataFrame valido per entrambi i modelli
dati_finali = pd.DataFrame([features_comuni])[ordine_colonne_finale]

# Convertiamo in tipo categorico standard le due colonne rimaste testuali per evitare vecchi errori di formato
dati_finali['Year_of_Study'] = dati_finali['Year_of_Study'].astype('category')
dati_finali['Prompt_Engineering_Skill'] = dati_finali['Prompt_Engineering_Skill'].astype('category')


# Checkbox di Debug per l'interfaccia di Streamlit
if st.checkbox("Mostra il DataFrame di input unificato inviato ai modelli (22 feature)"):
    st.dataframe(dati_finali)


# 5. PULSANTE DI CALCOLO UNICO E OUTPUT
st.write("") # Spazio vuoto
if st.button("🚀 Calcola Predizioni", type="primary", use_container_width=True):
    
    st.markdown("### 📊 Risultati dell'Analisi")
    
    # Creiamo due colonne per mostrare i risultati affiancati in modo elegante
    res_col1, res_col2 = st.columns(2)
    
    # --- COLONNA 1: PREDIZIONE GPA ---
    with res_col1:
        st.markdown("#### 🎯 Performance Accademica")
        if model_gpa is not None:
            pred_gpa = model_gpa.predict(dati_finali)[0] 
            st.metric(label="Post_Semester_GPA Predetto", value=f"{pred_gpa:.3f}")
        else:
            st.warning("⚠️ File 'gpa_model.pkl' non trovato. Mostro simulazione:")
            simulated_gpa = max(0.0, min(4.0, pre_gpa - (weekly_hours * 0.01) + (study_hours * 0.02)))
            st.metric(label="Post_Semester_GPA (Simulato)", value=f"{simulated_gpa:.3f}")

    # --- COLONNA 2: PREDIZIONE BURNOUT ---
    with res_col2:
        st.markdown("#### 🧠 Benessere dello Studente")
        if model_burnout is not None:
            # Passiamo lo stesso DataFrame a 22 colonne a LightGBM/VotingClassifier
            pred_burnout = model_burnout.predict(dati_finali)[0]
            
            # Colora l'output in base al risultato del modello reale
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