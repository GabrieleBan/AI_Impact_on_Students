import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def spiegazione_teorica():
    print("=" * 80)
    print(" 1. FILOSOFIA DI BASE: BAGGING vs BOOSTING")
    print("=" * 80)
    print(
        "RANDOM FOREST (Bagging - Bootstrap Aggregating):\n"
        "  - Approccio 'Democratico' e in parallelo.\n"
        "  - Crea N alberi decisionali indipendenti l'uno dall'altro.\n"
        "  - Ogni albero viene addestrato su una porzione casuale dei dati (Bootstrap)\n"
        "    e valuta solo un sottoinsieme casuale di feature (colonne) ad ogni split.\n"
        "  - Predizione finale: Voto di maggioranza (classificazione) o media (regressione).\n"
        "  - Obiettivo principale: Ridurre la VARIANZA (evitare l'overfitting di alberi complessi).\n"
    )
    print(
        "XGBOOST (Sequential Boosting - eXtreme Gradient Boosting):\n"
        "  - Approccio 'Meritocratico' e in sequenza.\n"
        "  - Gli alberi vengono costruiti uno dopo l'altro.\n"
        "  - Ogni nuovo albero non predice il target finale, ma tenta di predire l'ERRORE (residuo)\n"
        "    commesso dall'insieme degli alberi precedenti, ottimizzando una funzione di perdita\n"
        "    tramite la discesa del gradiente.\n"
        "  - Predizione finale: Somma pesata delle predizioni di tutti gli alberi intermedi.\n"
        "  - Obiettivo principale: Ridurre il BIAS (errore sistematico) senza soffrire di varianza,\n"
        "    grazie a una forte regolarizzazione matematica nativa (L1/L2).\n"
    )

def tabella_comparativa_iperparametri():
    print("=" * 80)
    print(" 2. CONFRONTO DIRETTO DEGLI IPERPARAMETRI (Scikit-Learn vs XGBoost API)")
    print("=" * 80)
    
    confronto = {
        "Concetto Concettuale": [
            "Numero di alberi",
            "Profondità massima albero",
            "Passo di aggiornamento (Shrinkage)",
            "Regolarizzazione (L1 / L2)",
            "Campioni per albero",
            "Feature per split"
        ],
        "Random Forest (Sklearn)": [
            "n_estimators (default: 100)",
            "max_depth (default: None, alberi profondi)",
            "Non applicabile (N/A)",
            "Non nativa (si controlla con potatura/min_samples)",
            "bootstrap=True / max_samples",
            "max_features (default: 'sqrt')"
        ],
        "XGBoost": [
            "n_estimators (default: 100)",
            "max_depth (default: 6, alberi superficiali)",
            "learning_rate / eta (default: 0.3)",
            "alpha (L1) / lambda (L2) (nativa e forte)",
            "subsample (default: 1)",
            "colsample_bytree / colsample_bylevel"
        ]
    }
    
    df = pd.DataFrame(confronto)
    print(df.to_string(index=False))
    print("\n")

def simulazione_pipeline_addestramento():
    print("=" * 80)
    print(" 3. SIMULAZIONE PRATICA DI ADDESTRAMENTO")
    print("=" * 80)
    
    # 1. Generazione di un dataset
    print("[1/4] Generazione del dataset sintetico (1000 campioni, 20 feature)...")
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=15, 
        n_redundant=5, 
        random_state=42
    )
    
    # CORRETTO: test_test_split modificato in test_size
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 2. Addestramento Random Forest
    print("\n[2/4] Inizializzazione e addestramento di Random Forest...")
    from sklearn.ensemble import RandomForestClassifier
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,       # Gli alberi crescono fino alla massima purezza
        random_state=42,
        n_jobs=-1             # Sfrutta tutti i core della CPU in parallelo
    )
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"--> Accuratezza Random Forest: {rf_acc:.4f}")
    
    # 3. Addestramento XGBoost
    print("\n[3/4] Inizializzazione e addestramento di XGBoost...")
    try:
        from xgboost import XGBClassifier
        
        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=4,         # Alberi volutamente meno profondi (weak learners)
            learning_rate=0.1,   # Controlla l'impatto di ogni nuovo albero sequenziale
            reg_lambda=1.0,      # Regolarizzazione L2 nativa
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        xgb_preds = xgb_model.predict(X_test)
        xgb_acc = accuracy_score(y_test, xgb_preds)
        print(f"--> Accuratezza XGBoost: {xgb_acc:.4f}")
    except ImportError:
        print("--> [AVVISO]: Libreria 'xgboost' non installata nel sistema corrente.")
        print("    Per testare questa sezione, esegui: pip install xgboost")

    # 4. Regole empiriche conclusive
    print("\n[4/4] LINEE GUIDA SULLA SCELTA:")
    print("-" * 40)
    print("Scegli RANDOM FOREST se:")
    print("  - Vuoi un modello 'Plug & Play' solido senza fare tuning dei parametri.")
    print("  - Il rischio di overfitting è la tua preoccupazione principale.")
    print("  - Hai bisogno di addestrare alberi complessi rapidamente in parallelo sulla CPU.")
    print("\nScegli XGBOOST se:")
    print("  - Cerchi la massima accuratezza predittiva possibile (competizioni, produzione avanzata).")
    print("  - Il dataset presenta relazioni complesse e sottili difficili da catturare.")
    print("  - Hai tempo/risorse per ottimizzare gli iperparametri (learning rate, depth, regolarizzazione).")
    print("  - Vuoi sfruttare il supporto nativo per GPU o la gestione automatica dei dati mancanti.")
    print("=" * 80)

if __name__ == "__main__":
    spiegazione_teorica()
    tabella_comparativa_iperparametri()
    
    # CORRETTO: Invocazione esplicita della pipeline di addestramento principale
    simulazione_pipeline_addestramento()