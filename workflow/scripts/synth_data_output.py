import numpy as np
from dataclasses import dataclass

@dataclass
class SynthDataOutput:
    synth_data: object
    generator_diagnostics: object
    runtime: object
    n_orig: int
    generator_algorithm: str
    epsilon: float
    delta: float

