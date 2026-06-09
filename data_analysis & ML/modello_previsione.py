import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. CARICAMENTO E PULIZIA DATI
nome_file = "ai_student_impact_dataset.csv"

if not os.path.exists(nome_file):
    print(f"ERRORE: File '{nome_file}' non trovato.")
else:
    df = pd.read_csv(nome_file)
    df.columns = df.columns.str.strip()

    # 2. MAPPING ORDINALE CORRETTO 

    mapping_skill = {'Beginner': 0, 'Intermediate': 1, 'Advanced': 2}
    df['Prompt_Engineering_Skill_Num'] = df['Prompt_Engineering_Skill'].map(mapping_skill)

    # 3. SELEZIONE DELLE REGOLE 
    features = [
        'Pre_Semester_GPA',             # Il punto di partenza dello studente
        'Weekly_GenAI_Hours', 
        'Anxiety_Level_During_Exams', 
        'Traditional_Study_Hours', 
        'Prompt_Engineering_Skill_Num'
    ]
    
    X = df[features]
    y = df['Post_Semester_GPA']

    # 4. DIVISIONE DEI DATI 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. CREAZIONE E ADDESTRAMENTO DEL MODELLO
    print("Addestramento del modello in corso...")
    modello = RandomForestRegressor(n_estimators=150, random_state=42)
    modello.fit(X_train, y_train)
    print("Addestramento completato!")

    # 6. VALUTAZIONE DELLE PERFORMANCE
    previsioni = modello.predict(X_test)
    
    mae = mean_absolute_error(y_test, previsioni)
    r2 = r2_score(y_test, previsioni)

    print("\n--- NUOVI RISULTATI DEL TEST ---")
    print(f"Errore Medio Assoluto (MAE): {mae:.2f}")
    print(f"Accuratezza del Modello (R² Score): {r2 * 100:.2f}%")

    # 7. PROVA PRATICA (Senza Warning sui Feature Names)
    # Creiamo il nuovo studente come un DataFrame con i nomi delle colonne corretti
    nuovo_studente = pd.DataFrame([[3.2, 25, 8, 10, 0]], columns=features)
    
    voto_previsto = modello.predict(nuovo_studente)
    print(f"\nPrevisione per un nuovo studente (Pre_GPA: 3.2, Uso Alto IA, Competenze Basse):")
    print(f"-> Il voto finale (GPA) previsto è: {voto_previsto[0]:.2f}")