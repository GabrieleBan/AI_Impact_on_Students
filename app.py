import streamlit as st
import pandas as pd
import numpy as np
import joblib
from app_utils.user_input_processing import process_user_data_for_burnout_model, process_user_data_for_gpa_model, reorder_colums_for_burnout
import lightgbm 
# Configurazione pagina
st.set_page_config(page_title="AI & Student Performance Analytics", layout="wide")

# Caricamento modelli con cache
@st.cache_resource
def load_models(path_model):
    try:
        model = joblib.load(path_model)
        return model
    except FileNotFoundError:
        return None

# Caricamento modelli
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

# INTERFACCIA GRAFICA (TITOLO)
st.title("🎓 Impatto della GenAI sulle Performance degli Studenti")
st.markdown("Inserisci i dati dello studente per calcolare il **GPA** atteso e del **Rischio di Burnout**.")

# INPUT DELL'UTENTE (Organizzato in colonne)
st.subheader("📝 Dati dello Studente")

col1, col2, col3 = st.columns(3)

with col1:
    major = col1.selectbox("Facoltà (Major_Category)", ["Humanities", "Medical", "Business", "STEM", "Arts"])
    year = col1.selectbox("Anno di Studio", ["Freshman", "Sophomore", "Junior", "Senior","Graduate"])
    pre_gpa = col1.number_input("GPA ultimo semestre", min_value=0.0, max_value=4.0, value=3.0, step=0.25)
    policy = col3.selectbox("Policy Istituzionale AI", ["Allowed_With_Citation", "Strict_Ban", "Actively_Encouraged"])
    

with col2:
    study_hours = col3.number_input("Ore Studio Tradizionale/Settimana", min_value=0.0, max_value=100.0, value=15.0,step=1.0)
    weekly_ai_hours = col1.number_input("Ore settimanali uso GenAI", min_value=0.0, max_value=50.0, value=10.0, step=1.0)
    use_case = col2.selectbox("Uso Principale AI", ["Copywriting/Drafting", "Ideation", "Summarizing_Reading", "Coding", "Research"])
    paid_sub = col2.checkbox("Abbonamento GenAI a pagamento?")

with col3:
    anxiety = col3.slider("Livello Ansia Esami (1-10)", 1, 10, 5)
    ai_dep = col3.slider("Dipendenza percepita da AI (1-5)", 1, 10, 3)
    prompt_skill = col2.selectbox("Skill Prompt Engineering", ["Beginner", "Intermediate", "Advanced"])
    tool_div = col2.slider("Numero Strumenti Usati", 1, 10, 3)



# inizializzazione retrieve dati da UI


features_comuni = process_user_data_for_gpa_model(major, year, pre_gpa, policy, study_hours, weekly_ai_hours, use_case, paid_sub, anxiety, ai_dep, prompt_skill, tool_div)



gpa_data = pd.DataFrame([features_comuni])

# Convertiamo in tipo categorico standard le due colonne rimaste testuali per evitare vecchi errori di formato
from app_utils.from_categorical_to_value import get_prompt_level_map
gpa_data['Prompt_Engineering_Skill'] = gpa_data['Prompt_Engineering_Skill'].map(get_prompt_level_map())


burnout_data=process_user_data_for_burnout_model(major, year, pre_gpa, policy, study_hours, weekly_ai_hours, use_case, paid_sub, anxiety, ai_dep, prompt_skill, tool_div)
burnout_data=pd.DataFrame([burnout_data])

# Checkbox di Debug per l'interfaccia di Streamlit
# if st.checkbox("Mostra il DataFrame di input unificato inviato ai modelli (22 feature)"):
#         st.dataframe(burnout_data)
# pulsante calcolo gpa
st.write("") # Spazio vuoto
if st.button("🚀 Calcola Predizioni", type="primary", use_container_width=True):
    
    st.markdown("### 📊 Risultati dell'Analisi")
    
    # due colonne oer risultati affiancati 
    res_col1, res_col2 = st.columns(2)
    
    # PREDIZIONE GPA 
    
    with res_col1:
        st.markdown("#### 🎯 Performance Accademica")
        if model_gpa is not None:
            pred_gpa = model_gpa.predict(gpa_data)[0] 
            st.metric(label="Post_Semester_GPA Predetto", value=f"{pred_gpa:.3f}")
        else:
            st.warning("⚠️ File 'gpa_model.pkl' non trovato. Mostro simulazione:")
            pred_gpa = max(0.0, min(4.0, pre_gpa - (weekly_ai_hours * 0.01) + (study_hours * 0.02)))
            st.metric(label="Post_Semester_GPA (Simulato)", value=f"{pred_gpa:.3f}")

    burnout_data["Post_Semester_GPA"]=float(pred_gpa)

    
    with res_col2:
        st.markdown("#### 🧠 Rischio burnout studente")
        if model_burnout is not None:
            # Passiamo lo stesso DataFrame a 22 colonne a LightGBM/VotingClassifier
            burnout_data=reorder_colums_for_burnout(burnout_data)
            pred_burnout = model_burnout.predict_proba(burnout_data)[0]


            st.error(f"🔥 Alto: {pred_burnout[2]*100:.2f}%")
            st.warning(f"⚠️ Medio: {pred_burnout[1]*100:.2f}%")
            st.success(f"✅ Basso: {pred_burnout[0]*100:.2f}%")


        else:
            st.warning("⚠️ File 'burnout_model.pkl' non trovato. Mostro simulazione:")
            rischio = "High" if (weekly_ai_hours > 20 and anxiety > 7) else "Medium" if anxiety > 4 else "Low"
            
            if rischio == "High":
                st.error(f"🔥 Livello Rischio Burnout (Simulato): {rischio}")
            elif rischio == "Medium":
                st.warning(f"⚠️ Livello Rischio Burnout (Simulato): {rischio}")
            else:
                st.success(f"✅ Livello Rischio Burnout (Simulato): {rischio}")
