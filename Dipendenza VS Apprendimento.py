import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. IMPOSTAZIONE PERCORSO FILE
nome_file = "ai_student_impact_dataset.csv"

if not os.path.exists(nome_file):
    print(f"ERRORE: Il file '{nome_file}' non è stato trovato.")
else:
    # Caricamento del dataset
    df = pd.read_csv(nome_file)

    # Pulizia nomi colonne da eventuali spazi
    df.columns = df.columns.str.strip()

    # Utilizziamo le colonne ESATTE del tuo dataset
    # X: Ore IA | Y: Dipendenza Percepita | Colore: Ritenzione Competenze
    
    # Impostiamo lo stile di Seaborn
    sns.set_theme(style="whitegrid")
    
    # Creiamo la figura principale
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # 2. CREAZIONE DELLO SCATTERPLOT CON LE COLONNE REALI
    # c= definisce la variabile che pilota il colore (Skill_Retention_Score)
    # cmap='coolwarm_r': Rosso = bassa ritenzione, Blu = alta ritenzione
    scatter = ax.scatter(
        df['Weekly_GenAI_Hours'], 
        df['Perceived_AI_Dependency'], 
        c=df['Skill_Retention_Score'], 
        cmap='coolwarm_r',      
        alpha=0.8, 
        edgecolors='w', 
        s=120                   
    )
    
    # barra del colore laterale 
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Skill_Retention_Score (Ritenzione delle Competenze)', fontsize=11, labelpad=10)
    
    # 3. AGGIUNGIAMO UNA LINEA DI REGRESSIONE (TREND GENERALE)
    sns.regplot(
        data=df, 
        x='Weekly_GenAI_Hours', 
        y='Perceived_AI_Dependency', 
        scatter=False, 
        color='black', 
        line_kws={"linewidth": 2, "linestyle": "--"},
        label='Trend della Dipendenza'
    )
    
    # Personalizzazione dei testi e dei titoli basati sulle tue colonne
    ax.set_title("Il Paradosso dell'IA: Dipendenza Percepita vs Apprendimento", fontsize=14, pad=15, fontweight='bold')
    ax.set_xlabel('Ore Settimanali di Uso IA (Weekly_GenAI_Hours)', fontsize=11)
    ax.set_ylabel('Dipendenza Percepita dall\'IA (Perceived_AI_Dependency)', fontsize=11)
    ax.legend(loc='upper left')
    
    # Ottimizza lo spazio e mostra il grafico
    plt.tight_layout()
    plt.show()