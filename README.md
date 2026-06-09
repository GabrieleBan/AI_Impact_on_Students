# AI impact on Students

Team:   
* Gabriele Banchio: https://www.linkedin.com/in/gabriele-banchio-6842113b5/
* Anna Kotelevych https://www.linkedin.com/in/anna-kotelevych-2089b02b1/
* Manuel Luis Cortes: https://www.linkedin.com/in/manuel-luis-cortes-7318b236a/
* Adriano Brancatello https://www.linkedin.com/in/adriano-brancatello/

Dataset utilizzato: https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students
Raccoglie informazioni relative agli studenti americani di diversi corsi di studio, tra cui GPA, ore dedicate a studio tradizionale o utilizzo di AI.

Riassunto:

Durante il lavoro di analisi del dataset abbiamo inseguito 2 obiettivi in parallelo:
- abbiamo studiato l'impatto dell'utilizzo dell'IA sul rendimento scolastico degli studenti scoprendo che un utilizzo contenuto alcuni studenti sono ingrado di poter migliorare i loro voti, ma che lo studio tradizionale è imprescindibile e un uso esagerato dell'IA porta a risultati controproducenti.
    
- abbiamo studiato linfluenza della IA sullo stato di burnout e stress accademico, con attenzione a come l'uso delle piattaforme di LLM influenza l'approccio al mondo accademico influenzando prestazioni e salute mentale

In seguito abbiamo cercato di fare leva sulle informazioni per migliorare le prestazioni dei modelli addestrati. 
I modelli addestrati predicono il GPA degli studenti e il rischio di burnout degli studenti.


Deploy:

- Installare le dipendenze con: pip install -r requirements.txt
- Per lanciare l'interfaccia grafica utilizzare il comando: streamlit run .\app.py

Struttura:
- app_utils/ contiene alcune funzioni necessarie al processing dell'input utente
- data_analysis & ML/ contiene i file jupyter notebook utilizzati durante l'analisi e alcuni risultati ottenuti
- 'PRESENTAZIONE PROGETTO.pptx' contiene una presentazione del lavoro svolto
