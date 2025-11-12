"""
cost_accuracy_analyzer.py - Analisi tradeoff costo/accuratezza per modelli LLM.

Questo modulo analizza il compromesso tra costi di utilizzo API e accuratezza
dei modelli di estrazione parametri da datasheet. Genera:
- CSV aggregato con metriche per modello
- Grafici scatter costo vs accuratezza
- Frontiera di Pareto per identificare modelli efficienti
- Bar chart delle accuratezze

Input: Predizioni in results/{model}/outputs/, ground truth in ground_truth/
Output: CSV e PNG in reports/
Dipendenze: matplotlib
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

RESULTS_DIR = Path("output/exp1/results")
GT_DIR = Path("output/ground_truth_jsons")
REPORTS = Path("output/exp1/reports")
REPORTS.mkdir(exist_ok=True)

NUM_TOL = 0.1  # tolleranza (mm) per campi numerici


def is_number(x):
    try:
        float(x)
        return True
    except Exception:
        return False


def field_equal(a: Any, b: Any, tol: float = NUM_TOL) -> bool:
    if a is None or b is None:
        return False
    if is_number(a) and is_number(b):
        return abs(float(a) - float(b)) <= tol
    return str(a).strip().lower() == str(b).strip().lower()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def model_cost_usd(model_dir: Path) -> float:
    total = 0.0
    usage_dir = model_dir / "usage"
    for csvf in usage_dir.glob("*.csv"):
        with csvf.open("r", encoding="utf-8") as fp:
            rdr = csv.reader(fp)
            next(rdr, None)
            for row in rdr:
                if not row or row[0].upper() not in ("TOTAL", "WALL"):
                    try:
                        total += float(row[3])
                    except Exception:
                        pass
    return total


def evaluate_model(model: str) -> Tuple[int, int, int, float, float]:
    model_dir = RESULTS_DIR / model
    pred_dir = model_dir / "outputs"
    files = list(pred_dir.glob("*.json"))
    fields_total = 0
    fields_correct = 0

    for p in files:
        gt = GT_DIR / p.name
        if not gt.exists():
            continue
        gtj = load_json(gt)
        pj = load_json(p)
        for k, v in gtj.items():
            fields_total += 1
            if field_equal(pj.get(k), v):
                fields_correct += 1

    files_count = len(files)
    acc = (fields_correct / fields_total) if fields_total else 0.0
    cost = model_cost_usd(model_dir)
    return files_count, fields_total, fields_correct, acc, cost


def pareto_front(points: List[Tuple[float, float, str]]):
    points = sorted(points)
    frontier: List[Tuple[float, float, str]] = []
    best_acc = -1
    for c, a, m in points:
        if a > best_acc:
            frontier.append((c, a, m))
            best_acc = a
    return frontier


def main():
    models = [d.name for d in RESULTS_DIR.iterdir() if d.is_dir()]
    rows = []
    points = []
    for m in models:
        files_count, ft, fc, acc, cost = evaluate_model(m)
        rows.append([m, files_count, ft, fc, round(acc, 4), round(cost, 4)])
        points.append((cost, acc, m))

    with (REPORTS / "agg_metrics.csv").open("w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["model", "files", "fields_total", "fields_correct", "accuracy", "cost_usd"])
        w.writerows(rows)

    plt.figure()
    for c, a, m in points:
        plt.scatter(c, a)
        plt.text(c, a, m)
    plt.xlabel("Costo totale (USD)")
    plt.ylabel("Accuracy")
    plt.title("Trade-off costo vs accuracy (per modello)")
    plt.tight_layout()
    plt.savefig(REPORTS / "cost_vs_accuracy.png", dpi=160)

    front = pareto_front(points)
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
        plt.savefig(REPORTS / "pareto_frontier.png", dpi=160)

    plt.figure()
    labels = [m for _, _, m in points]
    accs = [a for _, a, _ in points]
    plt.bar(labels, accs)
    plt.ylabel("Accuracy")
    plt.xticks(rotation=30, ha="right")
    plt.title("Accuracy per modello")
    plt.tight_layout()
    plt.savefig(REPORTS / "accuracy_bar.png", dpi=160)


if __name__ == "__main__":
    main()
