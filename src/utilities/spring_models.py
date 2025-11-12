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
            "Funzione della molla dedotta da titolo/tabella/testo e da caratteristiche visive.\n"
            "- compression: elicoidale senza bracci; presenza di L0, De/Di, indicazioni di passo e/o note “closed & ground”.\n"
            "- torsion: elicoidale con uno o due bracci; presenti quote angolari e lunghezze bracci (possono coesistere indicazioni di L0, De/Di).\n"
            "- not_supported: molla non tra i tipi gestiti (es. trazione/garter) → non proseguire con i campi “compression”.\n"
            "- not_a_spring: il file non raffigura un disegno tecnico o una tabella relativa a una molla."
        ),
    )

    # Metadati (validi per qualsiasi molla)
    wire_diameter: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Diametro del filo d; può essere espresso in mm oppure in pollici (inches).\n"
            "Dove cercare: simboli/etichette “d”, “wire Ø”, “⌀wire”, “Ø”, “wire dia” (in tabella o nelle note).\n"
            "Se non riportato ma De e Di sono chiaramente definiti → stimare come (De − Di) / 2.\n"
            "Non confondere con il diametro del corpo molla.\n"
            "Arrotondare a 0.01 mm dopo eventuale conversione da pollici (1\" = 25.4 mm)."
        )
    )
    wire_material: Optional[WireMaterial] = Field(
        None,
        description=(
            "Materiale del filo. Normalizzare i sinonimi più comuni:\n"
            "stainless_steel ← AISI 302/304/316, EN 10270-3 (NS), X10CrNi…\n"
            "chrome_silicon_steel ← CrSi, Si-Cr, EN 10270-2, ASTM A401\n"
            "music_wire_steel ← Music wire, EN 10270-1 (SH/DM), ASTM A228\n"
            "Se assente o ambiguo → null. (Esistono altre varianti più rare: per ora non considerate.)"
        )
    )

    # Parametri comuni alle molle a compressione
    spring_type: Optional[SpringType] = Field(
        None,
        description=(
            "Famiglia geometrica (solo se spring_function = compression):\n"
            "cylindrical: De ~ costante lungo l’asse (variazioni ≤ 2–3%).\n"
            "conical: De varia monotonicamente (cono o tronco di cono).\n"
            "biconical simmetrica: restringimento centrale (clessidra) con De_min al centro oppure bombatura a “botte” con De_max al centro.\n"
            "biconical asimmetrica: tre diametri distinti; D1 iniziale (quello di taglio memorizzato in manuale), D2 centrale (può essere maggiore o minore di D1 e D3), D3 finale diverso da D1 (tipicamente più grande).\n"
            "custom: profilo non riconducibile ai casi precedenti (sezioni miste, raccordi multipli, richieste speciali di avvolgimento)."
        )
    )
    free_length: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Lunghezza libera L0 a molla non caricata, misurata lungo l’asse. Escludere eventuali elementi di estremità non elicoidali (ganci, staffe).\n"
            "Nel PG si inserisce la lunghezza libera di molle non molate.\n"
            "Se la molla è molata (note “estremità molate”, “closed ends ground”, ecc.), la L0 riportata a disegno va corretta:\n"
            "L0_PG = L0_disegno + 1.5 · Ø (Ø = diametro filo)."
        )
    )
    total_coils: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Numero totale di spire Na (incluse spire chiuse e porzioni di spira).\n"
            "Accetta frazioni fino al secondo decimale (es. 7.25).\n"
            "Se il disegno indica “closed & ground”, le spire chiuse contano nel totale."
        )
    )
    initial_closed_coils: Optional[int] = Field(
        None,
        ge=0,
        description=(
            "Spire chiuse (passo ≈ 0) all’inizio (lato A).\n"
            "Indizi: note “closed/ground end”, tratteggio ad aderenza, simbolo di passo nullo.\n"
            "Valore tipico 1; può essere > 1 (es. 1.5, 2, …). Spesso il disegno riporta Na e working coils: in tal caso le spire chiuse si ottengono per differenza/2."
        )
    )
    final_closed_coils: Optional[int] = Field(
        None,
        ge=0,
        description=(
            "Spire chiuse (passo ≈ 0) alla fine (lato B).\n"
            "Se un solo lato è chiuso, l’altro = 0."
        )
    )
    pitch_insertion_coils: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Numero di spire con passo crescente (zona di inserzione).\n"
            "Identificare porzioni dove la distanza tra spire aumenta progressivamente; ammesse frazioni.\n"
            "Di solito non è riportato: impostare 1 salvo evidenze diverse dal disegno."
        )
    )
    pitch_retraction_coils: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Numero di spire con passo decrescente (zona di retrazione).\n"
            "Identificare porzioni dove la distanza tra spire diminuisce progressivamente; ammesse frazioni."
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
            "Diametro esterno De/Do/Øe nominale (mm) del corpo molla (in inglese OD).\n"
            "Se in tabella compaiono De e Di, usare De.\n"
            "Non includere allargamenti localizzati dovuti a lavorazioni di estremità."
        )
    )
    body_diameter_correction: Optional[float] = Field(
        None,
        description=(
            "Correzione ΔD applicata al diametro corpo per compensazioni di processo (es. rilassamento, grinding).\n"
            "Positiva se il corpo risultante è più grande, negativa se più piccolo.\n"
            "Sul disegno è raramente indicata ma in pratica si applica quasi sempre.\n"
            "Per abilitare la casella dedicata nel programma, inserire sempre un valore minimo: 0.01 (se non diversamente specificato)."
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
            "Diametro minimo (punta stretta) in mm. Cercare “Dmin”, “small end OD”. Con tolleranze, usare il nominale."
        )
    )
    maximum_diameter: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Diametro massimo (punta larga) in mm. Cercare “Dmax”, “large end OD”. Non confondere con diametri di flange o sedi."
        )
    )
    concavity_correction: Optional[float] = Field(
        None,
        description=(
            "Correzione di conicità (mm). Queste informazioni di solito non ci sono e generalmente si lascia 0; spesso dipende da verifiche post-prototipo (prove carico):\n"
            "> 0 → profilo convesso (apertura più rapida del lineare).\n"
            "< 0 → profilo concavo (apertura più lenta del lineare).\n"
            "= 0 → variazione diametro lineare.\n"
            "Se non applicabile/non deducibile → null."
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
            "Diametro all’estremità iniziale (lato A) in mm; simboli comuni: D1, De_in.\n"
            "Deve essere il più piccolo tra i (due o tre) diametri. È quello da cui si parte ad avvolgere in macchina.\n"
            "Usare il diametro del corpo molla in prossimità dell’inizio, non sedi/accessori."
        )
    )
    central_diameter: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Diametro centrale (zona di minima o massima sezione, a seconda che sia clessidra o botte) in mm; simboli: Dc, Dmin_central, Dmax_central.\n"
            "È la strozzatura/bombatura tipica del profilo biconico; nella pratica, le biconiche sono più spesso a botte che a clessidra."
        )
    )
    final_diameter: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Diametro all’estremità finale (lato B) in mm; simboli: D2, De_out, D3.\n"
            "Specificare il diametro del corpo in prossimità della fine, non di sedi o flange.\n"
            "Può essere uguale o diverso dal diametro iniziale."
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
            "Usare custom quando il profilo non è chiaramente cilindrico, conico o biconico (es.: sezioni miste, raccordi multipli, richieste speciali di avvolgimento)."
        )
    )
