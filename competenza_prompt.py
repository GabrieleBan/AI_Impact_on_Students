import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. IMPOSTAZIONE PERCORSO FILE
nome_file = "ai_student_impact_dataset.csv"

if not os.path.exists(nome_file):
    print(f"ERRORE: Il file '{nome_file}' non è stato trovato.")
else:
    df = pd.read_csv(nome_file)
    df.columns = df.columns.str.strip() 

    # Configurazione stile grafico
    sns.set_theme(style="white")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # ------------------------------------------------------------------
    # GRAFICO 1: densità di distribuzione e linee di trend 
    # ------------------------------------------------------------------
    ordine_skill = ['Beginner', 'Intermediate', 'Advanced']
    ordine_effettivo_skill = [s for s in ordine_skill if s in df['Prompt_Engineering_Skill'].unique()]

    # Creiamo una mappa di densità continua 
    sns.kdeplot(
        data=df,
        x='Weekly_GenAI_Hours',
        y='Post_Semester_GPA',
        hue='Prompt_Engineering_Skill',
        hue_order=ordine_effettivo_skill,
        palette='viridis',
        fill=True,         
        alpha=0.4,          
        levels=5,           
        thresh=0.2,        
        ax=ax1
    )
    
    # Sovrapponiamo solo le linee di trend pulite per evidenziare la direzione
    colori = sns.color_palette('viridis', len(ordine_effettivo_skill))
    for skill, colore in zip(ordine_effettivo_skill, colori):
        subset = df[df['Prompt_Engineering_Skill'] == skill]
        sns.regplot(
            data=subset,
            x='Weekly_GenAI_Hours',
            y='Post_Semester_GPA',
            scatter=False,  
            ax=ax1,
            color=colore,
            label=f'Trend {skill}',
            line_kws={"linewidth": 3}
        )

    ax1.set_title("Densità di Distribuzione e Trend dei Voti (Senza Punti)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Ore Settimanali IA (Weekly_GenAI_Hours)", fontsize=11)
    ax1.set_ylabel("Media Voti Finale (Post_Semester_GPA)", fontsize=11)
    ax1.legend(title="Prompt Skill")

    # ------------------------------------------------------------------
    # GRAFICO 2: l'istogramma del Burnout 
    # ------------------------------------------------------------------
    ordine_burnout = ['Low', 'Medium', 'High']
    ordine_effettivo_burnout = [b for b in ordine_burnout if b in df['Burnout_Risk_Level'].unique()]

    sns.countplot(
        data=df, 
        x='Burnout_Risk_Level', 
        hue='Prompt_Engineering_Skill', 
        order=ordine_effettivo_burnout,
        hue_order=ordine_effettivo_skill,
        palette='viridis',
        ax=ax2
    )

    ax2.set_title("Abilità nei Prompt vs Livello di Burnout dello Studente", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Rischio di Burnout (Burnout_Risk_Level)", fontsize=11)
    ax2.set_ylabel("Numero di Studenti", fontsize=11)
    ax2.legend(title="Prompt Skill")

    plt.tight_layout()
    plt.show()