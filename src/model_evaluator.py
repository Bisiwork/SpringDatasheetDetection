"""
model_evaluator.py - Modulo per valutazione e benchmarking dei modelli LLM.

Questo modulo fornisce funzionalità complete per la valutazione delle prestazioni
dei modelli di estrazione parametri da datasheet di molle. Include:
- Calcolo metriche accuratezza per modello (confronto predizioni vs ground truth)
- Aggregazione costi e tempi di esecuzione
- Generazione grafici: scatter costo/accuratezza, frontiera Pareto, bar charts
- Selezione best model basata su strategie multiple (utopia, acc_per_dollar, pareto)

Input: Predizioni JSON in results/{model}/outputs/, ground truth in ground_truth/
Output: CSV metriche aggregate in reports/, grafici PNG in reports/, best_model.json
Dipendenze: pandas, matplotlib
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

# ====== CONFIG ======
RESULTS_DIR = Path("output/exp1/results")  # es: results/<model>/{outputs,usage}
GT_DIR = Path("output/ground_truth_jsons")  # es: output/ground_truth_jsons/<file>.json
REPORTS_DIR = Path("output/exp1/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Tolleranze per numerici (mm)
NUM_TOL = 0.1


# ====== UTILS ======
def _is_number(x) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def _eq(a, b, tol: float = NUM_TOL) -> bool:
    if a is None or b is None:
        return False
    if _is_number(a) and _is_number(b):
        return abs(float(a) - float(b)) <= tol
    return str(a).strip().lower() == str(b).strip().lower()


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _aggregate_usage(model_dir: Path) -> Tuple[float, float, float]:
    """
    Ritorna (total_usd, api_time_secs, wall_time_secs)
    dai CSV in results/<model>/usage/*.csv
    """

    total_usd = 0.0
    api_secs = 0.0
    wall_secs = 0.0
    for csvf in (model_dir / "usage").glob("*.csv"):
        with csvf.open("r", encoding="utf-8") as fp:
            rdr = csv.reader(fp)
            header = next(rdr, None)
            for row in rdr:
                if not row:
                    continue
                tag = row[0].upper()
                if tag in ("TOTAL", "WALL"):
                    # riga di riepilogo (TOTAL EUR,secs) e (WALL ,,, ,wall_secs)
                    # sommiamo i USD dalle righe di stage, non da TOTAL
                    continue
                # Colonne: stage,in_tok,out_tok,USD,EUR,secs
                try:
                    total_usd += float(row[3] or 0.0)
                    api_secs += float(row[5] or 0.0)
                except Exception:
                    pass
            # seconda passata per leggere WALL
            fp.seek(0)
            rdr = csv.reader(fp)
            next(rdr, None)
            for row in rdr:
                if row and row[0].upper() == "WALL":
                    try:
                        wall_secs += float(row[-1] or 0.0)
                    except Exception:
                        pass
    return total_usd, api_secs, wall_secs


def evaluate_model(model: str) -> Dict[str, Any]:
    """
    Calcola metriche per un modello:
      - files, fields_total, fields_correct, accuracy
      - cost_usd, api_time_secs, wall_time_secs, avg_api_time_per_file
    """

    mdir = RESULTS_DIR / model
    pred_dir = mdir / "outputs"
    preds = list(pred_dir.glob("*.json"))
    fields_total = 0
    fields_correct = 0
    files_count = 0

    for p in preds:
        gt = GT_DIR / p.name
        if not gt.exists():
            continue
        pj = _load_json(p)
        gj = _load_json(gt)
        files_count += 1
        for k, v in gj.items():  # valutiamo sui campi annotati dal GT
            fields_total += 1
            if _eq(pj.get(k), v):
                fields_correct += 1

    acc = (fields_correct / fields_total) if fields_total else 0.0
    cost_usd, api_secs, wall_secs = _aggregate_usage(mdir)
    avg_api_time_per_file = (api_secs / files_count) if files_count else 0.0

    return dict(
        model=model,
        files=files_count,
        fields_total=fields_total,
        fields_correct=fields_correct,
        accuracy=acc,
        cost_usd=round(cost_usd, 6),
        api_time_secs=round(api_secs, 3),
        wall_time_secs=round(wall_secs, 3),
        avg_api_time_per_file=round(avg_api_time_per_file, 3),
    )


def collect_all_models() -> List[str]:
    return sorted([d.name for d in RESULTS_DIR.iterdir() if d.is_dir()])


def summarize_all(models: List[str] | None = None) -> pd.DataFrame:
    models = models or collect_all_models()
    rows = [evaluate_model(m) for m in models]
    df = pd.DataFrame(rows)
    out_csv = REPORTS_DIR / "agg_metrics.csv"
    df.to_csv(out_csv, index=False)
    return df


def pareto_front(points: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
    """
    points: [(cost, accuracy, model)] -> frontiera (crescente per costo, con accuracy non-dominata)
    """

    pts = sorted(points, key=lambda t: (t[0], -t[1]))
    frontier: List[Tuple[float, float, str]] = []
    best_acc = -1.0
    for c, a, m in pts:
        if a > best_acc:
            frontier.append((c, a, m))
            best_acc = a
    return frontier


def pick_best(df: pd.DataFrame, *, strategy: str = "utopia") -> Dict[str, Any]:
    """
    Selezione 'best model':
      - 'utopia': minima distanza euclidea al punto (cost_min, acc_max) su scale normalizzate [0,1]
      - 'acc_per_dollar': massimizza (accuracy / cost_usd)
      - 'pareto_first': se su Pareto, sceglie quello con acc più alto; altrimenti quello con
                        min distanza all'utopia all'interno del fronte.
    """

    df = df.copy()
    if df.empty:
        return {}

    # Normalizzazione 0-1
    cost_min, cost_max = df["cost_usd"].min(), df["cost_usd"].max()
    acc_min, acc_max = df["accuracy"].min(), df["accuracy"].max()
    df["cost_n"] = 0.0 if cost_max == cost_min else (df["cost_usd"] - cost_min) / (cost_max - cost_min)
    df["acc_n"] = 0.0 if acc_max == acc_min else (df["accuracy"] - acc_min) / (acc_max - acc_min)

    if strategy == "acc_per_dollar":
        df["score"] = df["accuracy"] / df["cost_usd"].clip(lower=1e-9)
        best = df.sort_values("score", ascending=False).iloc[0]
    elif strategy == "pareto_first":
        pts = list(zip(df["cost_usd"], df["accuracy"], df["model"]))
        F = pareto_front(pts)
        front_models = {m for _, _, m in F}
        cand = df[df["model"].isin(front_models)].copy()
        # dentro il fronte scegliamo acc più alto
        best = cand.sort_values(["accuracy", "cost_usd"], ascending=[False, True]).iloc[0]
    else:  # 'utopia'
        # distanza al punto (cost=0, acc=1) nello spazio normalizzato
        df["dist"] = ((df["cost_n"] - 0.0) ** 2 + (df["acc_n"] - 1.0) ** 2) ** 0.5
        best = df.sort_values(["dist", "cost_usd"], ascending=[True, True]).iloc[0]

    return best.to_dict()


def plot_charts(df: pd.DataFrame, outdir: Path = REPORTS_DIR) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # Scatter costo vs accuracy + etichette
    plt.figure()
    for _, r in df.iterrows():
        plt.scatter(r["cost_usd"], r["accuracy"])
        plt.text(r["cost_usd"], r["accuracy"], r["model"])
    plt.xlabel("Costo totale (USD)")
    plt.ylabel("Accuracy")
    plt.title("Trade-off costo vs accuracy (per modello)")
    plt.tight_layout()
    plt.savefig(outdir / "cost_vs_accuracy.png", dpi=160)

    # Frontiera di Pareto
    pts = list(zip(df["cost_usd"], df["accuracy"], df["model"]))
    front = pareto_front(pts)
    if len(front) >= 2:
        xs = [c for c, _, _ in front]
        ys = [a for _, a, _ in front]
        plt.figure()
        plt.plot(xs, ys, marker="o")
        for c, a, m in front:
            plt.text(c, a, m)
        plt.xlabel("Costo (USD)")
        plt.ylabel("Accuracy")
        plt.title("Frontiera di Pareto")
        plt.tight_layout()
        plt.savefig(outdir / "pareto_frontier.png", dpi=160)

    # Bar accuracy
    plt.figure()
    order = df.sort_values("accuracy", ascending=False)
    plt.bar(order["model"], order["accuracy"])
    plt.ylabel("Accuracy")
    plt.xticks(rotation=25, ha="right")
    plt.title("Accuracy per modello")
    plt.tight_layout()
    plt.savefig(outdir / "accuracy_bar.png", dpi=160)

    # Bar tempi totali
    plt.figure()
    order = df.sort_values("wall_time_secs", ascending=True)
    plt.bar(order["model"], order["wall_time_secs"])
    plt.ylabel("Wall time (s)")
    plt.xticks(rotation=25, ha="right")
    plt.title("Tempo totale per modello")
    plt.tight_layout()
    plt.savefig(outdir / "time_bar.png", dpi=160)

    # Bar tempi medi per file
    plt.figure()
    order = df.sort_values("avg_api_time_per_file", ascending=True)
    plt.bar(order["model"], order["avg_api_time_per_file"])
    plt.ylabel("Tempo medio API per file (s)")
    plt.xticks(rotation=25, ha="right")
    plt.title("Tempo medio API per file")
    plt.tight_layout()
    plt.savefig(outdir / "avg_time_per_file_bar.png", dpi=160)

    # Nuovo grafico: tempi min/avg/max per file per modello
    plt.figure()
    models = df["model"]
    mins = []
    avgs = []
    maxs = []
    for model in models:
        mdir = RESULTS_DIR / model
        times = []
        for csvf in (mdir / "usage").glob("*.csv"):
            with csvf.open("r", encoding="utf-8") as fp:
                rdr = csv.reader(fp)
                next(rdr, None)  # header
                for row in rdr:
                    if row and row[0].upper() not in ("TOTAL", "WALL"):
                        try:
                            times.append(float(row[5] or 0.0))  # secs column
                        except Exception:
                            pass
        if times:
            mins.append(min(times))
            avgs.append(sum(times) / len(times))
            maxs.append(max(times))
        else:
            mins.append(0)
            avgs.append(0)
            maxs.append(0)

    x = range(len(models))
    bars_min = plt.bar(x, mins, width=0.25, label='Min', align='center')
    bars_avg = plt.bar([i + 0.25 for i in x], avgs, width=0.25, label='Avg', align='center')
    bars_max = plt.bar([i + 0.5 for i in x], maxs, width=0.25, label='Max', align='center')

    # Aggiungi etichette ai valori sulle barre
    for bars in [bars_min, bars_avg, bars_max]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    plt.xlabel('Modello')
    plt.ylabel('Tempo API per file (s)')
    plt.title('Tempi min/avg/max per processare un file per modello')
    plt.xticks([i + 0.25 for i in x], models, rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "time_stats_per_file_bar.png", dpi=160)


def run_benchmark_report(models: List[str] | None = None, strategy: str = "utopia") -> Dict[str, Any]:
    df = summarize_all(models)
    plot_charts(df, REPORTS_DIR)
    best = pick_best(df, strategy=strategy)
    # salva riepilogo “best”
    (REPORTS_DIR / "best_model.json").write_text(json.dumps(best, indent=2))

    # Genera report MD con immagini min/max tempo per modello
    generate_time_report_md(df)

    print("BEST MODEL:", json.dumps(best, indent=2))
    return best


def generate_time_report_md(df: pd.DataFrame) -> None:
    """Genera un report MD con le immagini che hanno tempi min/max per ciascun modello."""
    md_lines = ["# Report Tempi di Elaborazione per Modello\n"]
    md_lines.append("Questo report mostra le immagini con tempi minimi e massimi di elaborazione per ciascun modello.\n")

    for model in df["model"]:
        md_lines.append(f"## Modello: {model}\n")
        mdir = RESULTS_DIR / model
        file_times = {}
        for csvf in (mdir / "usage").glob("*.csv"):
            stem = csvf.stem
            times = []
            with csvf.open("r", encoding="utf-8") as fp:
                rdr = csv.reader(fp)
                next(rdr, None)  # header
                for row in rdr:
                    if row and row[0].upper() not in ("TOTAL", "WALL"):
                        try:
                            times.append(float(row[5] or 0.0))
                        except Exception:
                            pass
            if times:
                file_times[stem] = sum(times)  # tempo totale per file

        if file_times:
            min_file = min(file_times, key=file_times.get)
            max_file = max(file_times, key=file_times.get)
            min_time = file_times[min_file]
            max_time = file_times[max_file]
            md_lines.append(f"- **Immagine con tempo minimo**: {min_file} ({min_time:.3f}s)\n")
            md_lines.append(f"- **Immagine con tempo massimo**: {max_file} ({max_time:.3f}s)\n")
        else:
            md_lines.append("- Nessun dato di tempo disponibile.\n")
        md_lines.append("\n")

    (REPORTS_DIR / "time_report.md").write_text("".join(md_lines), encoding="utf-8")
    print("Report MD generato: reports/time_report.md")
