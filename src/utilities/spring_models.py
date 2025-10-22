"""
spring_models.py - Modelli Pydantic per parametri tecnici di molle a compressione.

Questo modulo definisce i modelli dati strutturati per i parametri tecnici
estratte da datasheet di molle elicoidali a compressione. Utilizza Pydantic
per validazione e type hints, con descrizioni dettagliate per il prompt
engineering dei modelli LLM.

Classi principali:
- SpringBase: Campi comuni a tutte le molle
- CylindricalSpring: Molle cilindriche
- ConicalSpring: Molle coniche
- BiconicalSpring: Molle biconiche
- CustomSpring: Geometrie speciali

Ogni campo include descrizione tecnica per l'estrazione automatica.
Dipendenze: pydantic
"""
from enum import Enum
from typing import Optional, Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# ENUMS DI SUPPORTO
# --------------------------------------------------------------------------- #

class SpringFunction(str, Enum):
    """Classificazione funzionale o fuori ambito."""
    COMPRESSION   = "compression"     # Molla elicoidale a compressione (L0, De/Di, spire chiuse/passo)
    TORSION       = "torsion"         # Molla a torsione con bracci/leve; presenza di angoli/leg lengths
    NOT_SUPPORTED = "not_supported"   # Molla valida ma non tra i tipi gestiti (es. trazione/garter)
    NOT_A_SPRING  = "not_a_spring"    # Il documento/immagine non rappresenta una molla


class SpringType(str, Enum):
    """Famiglia geometrica (solo per molle a compressione)."""
    CYLINDRICAL = "cylindrical"  # Diametro esterno pressoché costante lungo l’asse
    CONICAL     = "conical"      # Diametro che varia monotonicamente (cono/cono rovescio)
    BICONICAL   = "biconical"    # Profilo “clessidra” (strozzatura centrale)
    CUSTOM      = "custom"       # Geometria non ricondotta ai tre casi precedenti


class WireMaterial(str, Enum):
    """Materiale del filo; mappa sinonimi tipici di datasheet."""
    STAINLESS_STEEL       = "stainless_steel"       # es. AISI 302/304/316, EN 10270-3-NS
    CHROME_SILICON_STEEL  = "chrome_silicon_steel"  # es. CrSi, EN 10270-2 (Si-Cr), ASTM A401
    MUSIC_WIRE_STEEL      = "music_wire_steel"      # es. C98/C100, EN 10270-1-SH, ASTM A228


ENUM_FIELD_MAP = {
    "spring_function": SpringFunction,
    "spring_type": SpringType,
    "wire_material": WireMaterial,
}


# --------------------------------------------------------------------------- #
# MODELLO BASE (Metadati + parametri comuni)
# --------------------------------------------------------------------------- #

class SpringBase(BaseModel):
    """
    Metadati e parametri estratti da disegni/FAQ/tabella:
    - Se più valori sono presenti (nominale/tolleranza), **riporta il nominale**.
    - Usa **mm** per tutte le quote lineari; arrotonda a **0.01 mm** salvo indicazioni diverse.
    - Se il valore non è chiaramente deducibile, imposta **null** (non 0).
    """
    spring_function: Optional[SpringFunction] = Field(
        None,
        description=(
            "Funzione della molla dedotta da: titolo/tabella/testo e caratteristiche visive.\n"
            "- compression: elicoidale senza bracci; presenza di L0, De/Di, passi/‘closed & ground’.\n"
            "- torsion: elicoidale con **due bracci**; quote angolari e lunghezze bracci.\n"
            "- not_supported: molla non tra i tipi gestiti (es. trazione/garter) → NON proseguire con campi ‘compression’.\n"
            "- not_a_spring: il file non raffigura una molla."
        ),
    )

    # Metadati (validi per qualsiasi molla)
    wire_diameter: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Diametro del filo **d** in mm. Dove cercare: simbolo ‘d’, ‘wire Ø’, ‘⌀wire’ in tabella.\n"
            "Se non riportato, stimare come (De − Di)/2 quando De/Di sono chiaramente definiti.\n"
            "Non confondere con **diametro del corpo**. Arrotonda a 0.01 mm."
        )
    )
    wire_material: Optional[WireMaterial] = Field(
        None,
        description=(
            "Materiale del filo. Normalizza sinonimi comuni:\n"
            " - stainless_steel ← AISI 302/304/316, EN 10270-3 (NS), X10CrNi…\n"
            " - chrome_silicon_steel ← CrSi, Si-Cr, EN 10270-2, ASTM A401\n"
            " - music_wire_steel ← music wire, EN 10270-1 (SH/DM), ASTM A228\n"
            "Se assente/ambiguo → null."
        )
    )

    # Parametri comuni alle molle a compressione
    spring_type: Optional[SpringType] = Field(
        None,
        description=(
            "Famiglia geometrica (solo se spring_function = compression):\n"
            " - cylindrical: De ~ costante lungo l’asse (variazioni ≤ 2–3%).\n"
            " - conical: De varia **monotonicamente** (cono o tronco di cono).\n"
            " - biconical: restringimento centrale (clessidra) con De_min al centro.\n"
            " - custom: profilo non riconducibile ai casi sopra (tabella con quote atipiche)."
        )
    )
    free_length: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Lunghezza **L0** a molla non caricata, misurata lungo l’asse.\n"
            "Escludere eventuali elementi di estremità non elicoidali (ganci, staffe)."
        )
    )
    total_coils: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Numero **totale** di spire **Na** (incluse **spire chiuse** e porzioni di spira). "
            "Accetta frazioni (es. 7.5). Se il disegno indica ‘closed & ground’, le spire chiuse "
            "contano nel totale."
        )
    )
    initial_closed_coils: Optional[int] = Field(
        None,
        ge=0,
        description=(
            "Spire **chiuse** (passo ≈ 0) all’inizio (**lato A**). "
            "Cerca note tipo ‘closed/ground end’, tratteggio ad aderenza, simbolo passo nullo."
        )
    )
    final_closed_coils: Optional[int] = Field(
        None,
        ge=0,
        description=(
            "Spire **chiuse** (passo ≈ 0) alla fine (**lato B**). "
            "Se un solo lato è chiuso, l’altro è 0."
        )
    )
    pitch_insertion_coils: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Numero di spire con **passo crescente** (zona di inserzione). "
            "Identifica porzioni dove la distanza tra spire aumenta progressivamente; "
            "accetta frazioni."
        )
    )
    pitch_retraction_coils: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Numero di spire con **passo decrescente** (zona di retrazione). "
            "Identifica porzioni dove la distanza tra spire diminuisce; accetta frazioni."
        )
    )


# --------------------------------------------------------------------------- #
# SOTTOCLASSI PER GEOMETRIE SPECIFICHE (compressione)
# --------------------------------------------------------------------------- #

class CylindricalSpring(SpringBase):
    """Molla cilindrica (De nominale costante)."""
    spring_type: Literal[SpringType.CYLINDRICAL] = Field(
        SpringType.CYLINDRICAL, description="Geometria: cilindrico (De ~ costante lungo l’asse)."
    )
    external_diameter: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Diametro **De/Do** esterno nominale (mm) del corpo molla. "
            "Se in tabella compaiono **De** e **Di**, usa **De**. "
            "Non includere eventuali allargamenti localizzati per lavorazioni di estremità."
        )
    )
    body_diameter_correction: Optional[float] = Field(
        None,
        description=(
            "Correzione **ΔD** applicata al diametro corpo per compensazioni di processo "
            "(es. rilassamento, grinding). Positiva se il corpo risultante è più grande, "
            "negativa se più piccolo. Lascia 0/null se non specificato."
        )
    )


class ConicalSpring(SpringBase):
    """Molla conica (diametro variabile monotono)."""
    spring_type: Literal[SpringType.CONICAL] = Field(
        SpringType.CONICAL, description="Geometria: conico (De varia monotonicamente)."
    )
    minimum_diameter: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Diametro **minimo** (punta stretta) in mm. Cerca ‘Dmin’, ‘small end OD’. "
            "Se presenti quote con tolleranza, prendi il **nominale**."
        )
    )
    maximum_diameter: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Diametro **massimo** (punta larga) in mm. Cerca ‘Dmax’, ‘large end OD’. "
            "Non confondere con diametri di flange o sedi."
        )
    )
    concavity_convexity: Optional[float] = Field(
        None,
        description=(
            "Indice di concavità/convessità complessiva del profilo conico (mm):\n"
            " - **> 0**: profilo **convesso** (apertura più rapida del previsto lineare)\n"
            " - **< 0**: profilo **concavo** (apertura più lenta)\n"
            " - **0**: variaz. diametro lineare. Lascia null se non applicabile."
        )
    )


class BiconicalSpring(SpringBase):
    """
    Molla biconica (clessidra). Diametri principali lungo l’asse e
    curvature locali per estremità/centro.
    """
    spring_type: Literal[SpringType.BICONICAL] = Field(
        SpringType.BICONICAL, description="Geometria: biconico (strozzatura centrale)."
    )

    initial_diameter: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Diametro all’estremità **iniziale** (lato A) in mm; simboli comuni: D1/De_in.\n"
            "Usa il diametro del corpo molla in prossimità dell’inizio, non eventuali sedi/accessori."
        )
    )
    central_diameter: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Diametro **centrale** (zona di minima sezione) in mm; simboli: Dc, Dmin_central.\n"
            "È la ‘strozzatura’ tipica del profilo biconico."
        )
    )
    final_diameter: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Diametro all’estremità **finale** (lato B) in mm; simboli: D2/De_out.\n"
            "Specifica il diametro del corpo in prossimità della fine, non di sedi o flange."
        )
    )
    initial_conical_coils: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Numero (anche frazionario) di spire coniche **verso l’inizio**. "
            "Riconosci la conicità locale da variazioni di De per spira."
        )
    )
    final_conical_coils: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Numero (anche frazionario) di spire coniche **verso la fine**. "
            "Se profilo simmetrico, i valori iniziale/finale tendono a coincidere."
        )
    )
    initial_coils_curvature: Optional[float] = Field(
        None,
        description=(
            "Curvatura locale (mm) delle prime spire rispetto a un profilo lineare equivalente:\n"
            " **> 0** convesso (diametro cresce più del lineare), **< 0** concavo."
        )
    )
    final_coils_curvature: Optional[float] = Field(
        None,
        description=(
            "Curvatura locale (mm) delle spire finali rispetto a un profilo lineare equivalente:\n"
            " **> 0** convesso, **< 0** concavo. Lascia null se non deducibile."
        )
    )


class CustomSpring(SpringBase):
    """Geometria speciale/non standard."""
    spring_type: Literal[SpringType.CUSTOM] = Field(
        SpringType.CUSTOM,
        description=(
            "Usa ‘custom’ quando il profilo non è chiaramente cilindrico, conico o biconico "
            "(ad es. sezioni miste, raccordi multipli, richieste speciali di avvolgimento)."
        )
    )
