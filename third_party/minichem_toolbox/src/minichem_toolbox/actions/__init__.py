"""Task-independent ActionSpec declarations grouped by scientific domain."""

from .scientific_data_interchange import ACTION_SPECS as SCIENTIFIC_DATA_INTERCHANGE_ACTIONS
from .structure_and_system import ACTION_SPECS as STRUCTURE_AND_SYSTEM_ACTIONS
from .cheminformatics import ACTION_SPECS as CHEMINFORMATICS_ACTIONS
from .format_conversion import ACTION_SPECS as FORMAT_CONVERSION_ACTIONS
from .molecular_electronic import ACTION_SPECS as MOLECULAR_ELECTRONIC_ACTIONS
from .reaction_and_kinetics import ACTION_SPECS as REACTION_AND_KINETICS_ACTIONS


ACTION_SPECS = (
    SCIENTIFIC_DATA_INTERCHANGE_ACTIONS
    + STRUCTURE_AND_SYSTEM_ACTIONS
    + CHEMINFORMATICS_ACTIONS
    + FORMAT_CONVERSION_ACTIONS
    + MOLECULAR_ELECTRONIC_ACTIONS
    + REACTION_AND_KINETICS_ACTIONS
)

__all__ = ["ACTION_SPECS"]
