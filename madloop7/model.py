from madloop7.madgraph_matrix_elements.gg_gddx.model.parameters import ModelParameters
from madloop7 import MadLoop7Error


class Parameter(object):
    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value
        self.real = value

    def lower(self) -> str:
        return self.name.lower()


class Particle(object):
    def __init__(self, name: str, mass: str = "zero", width: str = "zero"):
        self.name = name
        self.mass = mass
        self.width = width

    def get(self, characteristics: str) -> str:
        match characteristics:
            case 'mass': return self.mass
            case 'width': return self.width
            case _: raise MadLoop7Error(f"Particle characteristics {characteristics} not implemented")


class ModelInformation(object):

    def __init__(self, model_parameter: ModelParameters):
        self.parameters = model_parameter
        self.parameters_dict = {k: Parameter(k, v)
                                for k, v in self.parameters.__dict__.items()}

    def get_particle(self, pdg: int):

        match abs(pdg):
            case 1: p = Particle('d')
            case 2: p = Particle('u')
            case 3: p = Particle('s')
            case 4: p = Particle('c')
            case 5: p = Particle('b', "mdl_MB")
            case 6: p = Particle('t', "mdl_MT")
            case 11: p = Particle('e-')
            case 12: p = Particle('ve')
            case 13: p = Particle('mu-')
            case 14: p = Particle('vm')
            case 15: p = Particle('ta-', "mdl_MTA")
            case 16: p = Particle('vt')
            case 21: p = Particle('g')
            case 22: p = Particle('a')
            case 23: p = Particle('z', "mdl_MZ", "mdl_WZ")
            case 24: p = Particle('w+', "mdl_MW", "mdl_WW")
            case 25: p = Particle('h', "mdl_MH", "mdl_WH")
            case _: raise MadLoop7Error(f"Particle {pdg} not implemented")

        if pdg < 0:
            match p.name[-1]:
                case '-': p.name = p.name[:-1] + '+'
                case '+': p.name = p.name[:-1] + '-'
                case _: p.name = p.name + '~'

        return p

    def get(self, characteristics: str):
        match characteristics:
            case 'parameter_dict': return self.parameters_dict
            case _: raise MadLoop7Error(f"Model characteristics {characteristics} not implemented")
