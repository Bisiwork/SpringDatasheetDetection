# Spring Datasheet Detection

Utility per eseguire benchmark dei modelli e preparare il labeling dei dati di riferimento.

## Requisiti

Installa le dipendenze Python:

```bash
pip install -r requirements.txt
```

Su macOS è necessario anche:

```bash
brew install poppler
```

Questo pacchetto fornisce i binari richiesti da `pdf2image` per convertire i PDF in immagini.

## Benchmark e Labeling

- `python benchmark_models.py` esegue il flusso di estrazione per i modelli configurati.
- `python benchmark_models.py --report` genera `reports/agg_metrics.csv`, i grafici in `reports/` e stampa il best model calcolato da `src.benchmark_eval`.
- `python benchmark_models.py --labeling <model>` crea `reports/ground_truth_labeling.xlsx` utilizzando le predizioni del modello indicato.

I risultati prodotti (cartella `results/`) e i report (`reports/`) sono ignorati da Git.
