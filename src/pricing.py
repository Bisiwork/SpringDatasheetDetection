from __future__ import annotations
import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Optional, Any

class Pricing:
    def __init__(self, csv_path: str | Path, category: str = "Text tokens - Standard"):
        self.csv_path = Path(csv_path)
        self.category = category
        # Nota: cached_in può essere assente -> Optional[Decimal]
        self.table: Dict[str, Dict[str, Optional[Decimal]]] = {}
        self._load()

    @staticmethod
    def _usd_per_mtok_to_decimal(cell: Optional[str]) -> Optional[Decimal]:
        """
        Converte: '$1.25' -> Decimal('1.25').
        Ritorna None per '-', '', None.
        """
        if cell is None:
            return None
        s = cell.strip().replace(",", "")
        if not s or s == "-":
            return None
        if s.startswith("$"):
            s = s[1:]
        try:
            return Decimal(s)
        except InvalidOperation as e:
            raise ValueError(f"Valore prezzo non valido nel CSV: {cell!r}") from e

    def _load(self):
        with self.csv_path.open("r", encoding="utf-8") as fp:
            rdr = csv.DictReader(fp)
            for row in rdr:
                if row.get("Category", "").strip() != self.category:
                    continue
                model = row.get("Model", "").strip()
                if not model:
                    continue
                # Gestione robusta della colonna "Cached Input" (case-varianti)
                cached_raw = (
                    row.get("Cached Input")
                    or row.get("Cached input")
                    or row.get("Cached")
                    or row.get("CachedInput")
                )

                self.table[model] = {
                    "in": self._usd_per_mtok_to_decimal(row.get("Input")),
                    "cached_in": self._usd_per_mtok_to_decimal(cached_raw),
                    "out": self._usd_per_mtok_to_decimal(row.get("Output")),
                }

    def get_rates(self, model: str) -> Dict[str, Optional[Decimal]]:
        if model not in self.table:
            # Suggerimento: verifica di aver messo il modello nella *stessa Category*
            # del CSV che stai usando (es. 'Text tokens - Standard' per gpt-4o-mini).
            cats = sorted({self.category})
            raise KeyError(
                f"Model '{model}' non presente nella categoria '{self.category}' del CSV. "
                f"Controlla il CSV o la category selezionata."
            )
        rates = self.table[model]
        if rates["in"] is None or rates["out"] is None:
            raise ValueError(
                f"Prezzi incompleti per '{model}' in '{self.category}': {rates}. "
                f"Assicurati che 'Input' e 'Output' siano valorizzati."
            )
        return rates

    def cost_usd(self, model: str, usage: Any) -> float:
        """
        cost = uncached_in/1e6 * in + cached_in/1e6 * cached_price + out/1e6 * out
        Se 'cached_in' nel CSV è assente (None), usa il prezzo 'in' (nessuno sconto specifico).
        Prezzi sono per **1M token**. I cached tokens sono esposti in
        usage.input_tokens_details.cached_tokens (quando applicabile). :contentReference[oaicite:1]{index=1}
        """
        rates = self.get_rates(model)
        in_tok = int(getattr(usage, "input_tokens", 0))
        out_tok = int(getattr(usage, "output_tokens", 0))
        # supporta anche dict usage
        itd = getattr(usage, "input_tokens_details", None) or {}
        cached_tok = 0
        try:
            cached_tok = int(
                getattr(itd, "cached_tokens", None)
                if not isinstance(itd, dict) else itd.get("cached_tokens", 0)
            )
        except (TypeError, ValueError):
            cached_tok = 0

        cached_tok = max(0, min(cached_tok, in_tok))
        uncached_tok = in_tok - cached_tok

        price_in = rates["in"]              # garantito non None in get_rates()
        price_cached = rates["cached_in"] or price_in
        price_out = rates["out"]            # garantito non None

        usd = (Decimal(uncached_tok) / Decimal(1_000_000)) * price_in \
            + (Decimal(cached_tok)   / Decimal(1_000_000)) * price_cached \
            + (Decimal(out_tok)      / Decimal(1_000_000)) * price_out
        return float(usd)
