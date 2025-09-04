from enum import Enum
from typing import Optional, Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# ENUMS DI SUPPORTO
# --------------------------------------------------------------------------- #

class SpringFunction(str, Enum):
    """Tipo di funzionamento della molla (o classificazione fuori ambito)."""
    COMPRESSION   = "compression"     # molla a compressione
    TORSION       = "torsion"         # molla a torsione
    NOT_SUPPORTED = "not_supported"   # molla non supportata
    NOT_A_SPRING  = "not_a_spring"    # l’oggetto non è una molla


class SpringType(str, Enum):
    """Famiglia geometrica della molla (solo per molle a compressione)."""
    CYLINDRICAL = "cylindrical"
    CONICAL = "conical"
    BICONICAL = "biconical"
    CUSTOM = "custom"


class WireMaterial(str, Enum):
    """Materiale del filo della molla."""
    STAINLESS_STEEL = "stainless_steel"            # acciaio inox
    CHROME_SILICON_STEEL = "chrome_silicon_steel"  # acciaio al cromo-silicio
    MUSIC_WIRE_STEEL = "music_wire_steel"          # acciaio armonico


# --------------------------------------------------------------------------- #
# MODELLO BASE (Metadati validi per qualsiasi molla)
# --------------------------------------------------------------------------- #

class SpringBase(BaseModel):
    """
    Metadati e parametri comuni.
    NOTA: molte proprietà sono opzionali per consentire il riuso del modello
    anche per workflow “parziali” o con estrazione progressiva.
    """
    spring_function: Optional[SpringFunction] = Field(
        None,
        description=(
            "Tipo di funzionamento: compression | torsion | not_supported | not_a_spring"
        ),
    )

    # Metadati (validi per qualsiasi molla)
    wire_diameter: Optional[float] = Field(
        None, description="Diametro del filo impiegato in mm"
    )
    wire_material: Optional[WireMaterial] = Field(
        None, description="Materiale del filo (acciaio inox, cromo-silicio, armonico)"
    )

    # Parametri comuni alle molle a compressione
    spring_type: Optional[SpringType] = Field(
        None, description="Famiglia geometrica (cilindrico, conico, biconico, personalizzato)"
    )
    free_length: Optional[float] = Field(
        None, description="Lunghezza libera della molla non caricata in mm"
    )
    total_coils: Optional[float] = Field(
        None, description="Numero complessivo di spire"
    )
    initial_closed_coils: Optional[int] = Field(
        None, description="Spire chiuse all'inizio"
    )
    final_closed_coils: Optional[int] = Field(
        None, description="Spire chiuse alla fine"
    )
    pitch_insertion_coils: Optional[float] = Field(
        None, description="Numero di spire a passo crescente (fase di inserzione)"
    )
    pitch_retraction_coils: Optional[float] = Field(
        None, description="Numero di spire a passo decrescente (fase di retrazione)"
    )


# --------------------------------------------------------------------------- #
# SOTTOCLASSI PER GEOMETRIE SPECIFICHE (molle a compressione)
# --------------------------------------------------------------------------- #

class CylindricalSpring(SpringBase):
    """Molla cilindrica a diametro costante."""
    spring_type: Literal[SpringType.CYLINDRICAL] = Field(
        SpringType.CYLINDRICAL, description="Famiglia geometrica: cilindrico"
    )
    external_diameter: Optional[float] = Field(
        None, description="Diametro esterno costante lungo tutta la molla (mm)"
    )
    body_diameter_correction: Optional[float] = Field(
        None, description="Correzione applicata al diametro del corpo molla (mm)"
    )


class ConicalSpring(SpringBase):
    """Molla conica (tapered) a diametro variabile."""
    spring_type: Literal[SpringType.CONICAL] = Field(
        SpringType.CONICAL, description="Famiglia geometrica: conico"
    )
    minimum_diameter: Optional[float] = Field(
        None, description="Diametro minimo (punta stretta) in mm"
    )
    maximum_diameter: Optional[float] = Field(
        None, description="Diametro massimo (punta larga) in mm"
    )
    concavity_convexity: Optional[float] = Field(
        None, description="Indice di concavità/convessità complessiva (mm)"
    )


class BiconicalSpring(SpringBase):
    """Molla biconica (a clessidra)."""
    spring_type: Literal[SpringType.BICONICAL] = Field(
        SpringType.BICONICAL, description="Famiglia geometrica: biconico"
    )
    initial_diameter: Optional[float] = Field(
        None, description="Diametro delle prime spire (mm)"
    )
    central_diameter: Optional[float] = Field(
        None, description="Diametro nella zona centrale (min/max) in mm"
    )
    final_diameter: Optional[float] = Field(
        None, description="Diametro delle spire finali (mm)"
    )
    initial_conical_coils: Optional[float] = Field(
        None, description="Numero di spire coniche all'inizio"
    )
    final_conical_coils: Optional[float] = Field(
        None, description="Numero di spire coniche alla fine"
    )
    initial_coils_curvature: Optional[float] = Field(
        None, description="Curvatura concava/convessa delle prime spire (mm)"
    )
    final_coils_curvature: Optional[float] = Field(
        None, description="Curvatura concava/convessa delle spire finali (mm)"
    )


class CustomSpring(SpringBase):
    """Molla con parametri non standard (geometria speciale)."""
    spring_type: Literal[SpringType.CUSTOM] = Field(
        SpringType.CUSTOM, description="Famiglia geometrica: personalizzato"
    )
