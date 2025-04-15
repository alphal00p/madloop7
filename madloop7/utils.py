from enum import StrEnum
import logging
import madloop7.model
from . import logger
from madloop7 import MadLoop7Error
from madloop7.madgraph_matrix_elements.gg_gddx.model.parameters import ModelParameters
from madloop7.madgraph_matrix_elements.gg_gddx.phase_space_generator.flat_phase_space_generator import FlatInvertiblePhasespace
from madloop7.model import ModelInformation


class FlatPhaseSpaceGenerator(FlatInvertiblePhasespace):
    pass


class Colour(StrEnum):
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format=f'{Colour.GREEN}%(levelname)s{Colour.END} {Colour.BLUE}%(funcName)s l.%(lineno)d{
            Colour.END} {Colour.CYAN}t=%(asctime)s.%(msecs)03d{Colour.END} > %(message)s',
        datefmt='%Y-%m-%d,%H:%M:%S'
    )
    logger.setLevel(level)


def get_model(model="sm"):
    model = model.split('-')[0]
    match model:
        case "sm":
            model_params = ModelParameters(None)
            return ModelInformation(model_params)
        case _:
            raise MadLoop7Error(f"Model {model} not implemented")
