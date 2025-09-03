from enum import Enum
from typing import Optional, Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# SUPPORTING ENUMS
# --------------------------------------------------------------------------- #

class SpringType(str, Enum):
    """Geometric family of the spring."""
    CYLINDRICAL = "cylindrical"
    CONICAL = "conical"
    BICONICAL = "biconical"
    CUSTOM = "custom"


class WireMaterial(str, Enum):
    """Material of the spring wire."""
    STAINLESS_STEEL = "stainless_steel"
    CHROME_SILICON_STEEL = "chrome_silicon_steel"
    MUSIC_WIRE_STEEL = "music_wire_steel"

# --------------------------------------------------------------------------- #
# COMMON BASE MODEL FOR ALL SPRINGS
# --------------------------------------------------------------------------- #

class SpringBase(BaseModel):
    """
    Common parameters for all springs.
    Only spring_type is required.
    """
    spring_type: SpringType = Field(..., description="Geometric family of the spring")
    wire_material: Optional[WireMaterial] = Field(
        None, description="Material of the spring wire"
    )
    wire_diameter: Optional[float] = Field(
        None, description="Wire diameter in mm"
    )
    free_length: Optional[float] = Field(
        None, description="Free length of the spring in mm (unloaded)"
    )
    total_coils: Optional[float] = Field(
        None, description="Total number of coils"
    )
    initial_closed_coils: Optional[int] = Field(
        None, description="Number of closed coils at start"
    )
    final_closed_coils: Optional[int] = Field(
        None, description="Number of closed coils at end"
    )
    pitch_insertion_coils: Optional[float] = Field(
        None, description="Number of coils with increasing pitch (insertion)"
    )
    pitch_retraction_coils: Optional[float] = Field(
        None, description="Number of coils with decreasing pitch (retraction)"
    )


# --------------------------------------------------------------------------- #
# SUBCLASSES FOR SPECIFIC GEOMETRIES
# --------------------------------------------------------------------------- #

class CylindricalSpring(SpringBase):
    """Cylindrical spring with constant diameter."""
    spring_type: Literal[SpringType.CYLINDRICAL] = Field(
        SpringType.CYLINDRICAL, description="Geometric family: cylindrical"
    )
    external_diameter: Optional[float] = Field(
        None, description="Constant external diameter of the spring in mm"
    )
    body_diameter_correction: Optional[float] = Field(
        None, description="Diameter correction to apply to the body in mm"
    )


class ConicalSpring(SpringBase):
    """Conical (tapered) spring with variable diameter."""
    spring_type: Literal[SpringType.CONICAL] = Field(
        SpringType.CONICAL, description="Geometric family: conical"
    )
    minimum_diameter: Optional[float] = Field(
        None, description="Minimum (narrow end) diameter in mm"
    )
    maximum_diameter: Optional[float] = Field(
        None, description="Maximum (wide end) diameter in mm"
    )
    concavity_convexity: Optional[float] = Field(
        None, description="Overall concavity/convexity value in mm"
    )


class BiconicalSpring(SpringBase):
    """Biconical (hourglass) spring."""
    spring_type: Literal[SpringType.BICONICAL] = Field(
        SpringType.BICONICAL, description="Geometric family: biconical"
    )
    initial_diameter: Optional[float] = Field(
        None, description="Diameter at the first end in mm"
    )
    central_diameter: Optional[float] = Field(
        None, description="Diameter at the central section in mm"
    )
    final_diameter: Optional[float] = Field(
        None, description="Diameter at the second end in mm"
    )
    initial_conical_coils: Optional[float] = Field(
        None, description="Number of initial conical coils"
    )
    final_conical_coils: Optional[float] = Field(
        None, description="Number of final conical coils"
    )
    initial_coils_curvature: Optional[float] = Field(
        None, description="Curvature (concave/convex) of initial coils in mm"
    )
    final_coils_curvature: Optional[float] = Field(
        None, description="Curvature (concave/convex) of final coils in mm"
    )


class CustomSpring(SpringBase):
    """Spring with non-standard parameters (special geometry)."""
    spring_type: Literal[SpringType.CUSTOM] = Field(
        SpringType.CUSTOM, description="Geometric family: custom"
    )
    note: Optional[str] = Field(
        None, description="Free-form notes for special geometries or parameters"
    )


# --------------------------------------------------------------------------- #
# MACHINE PROGRAM ROW MODEL
# --------------------------------------------------------------------------- #

class SpringProgramRow(BaseModel):
    """
    Instruction row for CNC/winding machine: describes the movement
    needed to form the spring step-by-step.
    All fields are optional.
    """
    step: Optional[int] = Field(
        None, description="Step index (starting from 1)"
    )
    feed: Optional[float] = Field(
        None, description="Feed increment in coils"
    )
    feed_abs: Optional[float] = Field(
        None, description="Cumulative feed in coils"
    )
    diameter: Optional[float] = Field(
        None, description="Wire diameter at current step in mm"
    )
    rotation: Optional[float] = Field(
        None, description="Tool or wire rotation in degrees"
    )
    v_pitch: Optional[float] = Field(
        None, description="Vertical pitch in mm"
    )
    h_pitch: Optional[float] = Field(
        None, description="Horizontal pitch in mm"
    )
    mand_y: Optional[float] = Field(
        None, description="Y-axis carriage movement in mm"
    )
    mand_z: Optional[float] = Field(
        None, description="Z-axis carriage movement in mm"
    )
    funct: Optional[int] = Field(
        None, description="Machine function code executed at this step"
    )
    param: Optional[int] = Field(
        None, description="Optional parameter for the machine function"
    )
    speed: Optional[int] = Field(
        None, description="Machine speed percentage"
    )
