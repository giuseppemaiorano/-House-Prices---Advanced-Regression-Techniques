# House Prices - Advanced Regression Techniques

Questo progetto utilizza tecniche avanzate di regressione per predire i prezzi delle case. Utilizzando un dataset di Kaggle, il notebook esegue diverse analisi di preprocessamento e modellazione per ottenere previsioni accurate.

## Contenuto del Notebook

Il notebook contiene le seguenti sezioni principali:

1. **Introduzione**
    - Descrizione dell'obiettivo del progetto.
    - Panoramica del dataset utilizzato.

2. **Caricamento dei Dati**
    - Importazione delle librerie necessarie.
    - Caricamento del dataset e visualizzazione delle prime righe.

3. **Pulizia dei Dati**
    - Trattamento dei valori mancanti.
    - Conversione dei tipi di dati, se necessario.

4. **Preprocessamento dei Dati**
    - Creazione di pipeline per il preprocessamento dei dati numerici e categoriali.
    - Combinazione delle pipeline in un unico trasformatore.

5. **Addestramento del Modello**
    - Suddivisione del dataset in set di training e validation.
    - Addestramento del modello Random Forest Regressor.

6. **Valutazione del Modello**
    - Predizione sui dati di validation.
    - Calcolo della Root Mean Squared Error (RMSE).

7. **Predizioni sul Test Set**
    - Generazione delle predizioni sul dataset di test.
    - Creazione del file `submission.csv`.


## Requisiti

Per eseguire il notebook, è necessario avere installati i seguenti pacchetti Python:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Puoi installare questi pacchetti utilizzando il seguente comando:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
