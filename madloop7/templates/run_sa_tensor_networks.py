from phase_space_generator import FlatInvertiblePhasespace as PSGen
from phase_space_generator import LorentzVector, LorentzVectorList
import sys
import os
import random
import time

pjoin = os.path.join

# SYMBOLICA_COMMUNITY_PATH = None
SYMBOLICA_COMMUNITY_PATH = "%(symbolica_community_path)s"

if SYMBOLICA_COMMUNITY_PATH is not None:
    sc_path = os.path.abspath(
        pjoin(SYMBOLICA_COMMUNITY_PATH, "python"))
    if sc_path not in sys.path:
        sys.path.insert(0, sc_path)
    try:
        import symbolica_community as sc  # type: ignore # nopep8
    except:
        raise Exception('\n'.join([
            "ERROR: Could not import Python's symbolica_community module from {}".format(
                SYMBOLICA_COMMUNITY_PATH),
            "Add '<SYMBOLICA_COMMUNITY_INSTALLATION_DIRECTORY>/python' to your PYTHONPATH or specify the value of 'SYMBOLICA_COMMUNITY_PATH' in this standalone Python runner.",]))
else:
    try:
        import symbolica_community as sc  # type: ignore # nopep8
    except:
        raise Exception('\n'.join([
            "ERROR: Could not import Python's symbolica_community module.",
            "Add '<SYMBOLICA_COMMUNITY_INSTALLATION_DIRECTORY>/python' to your PYTHONPATH or specify the value of 'SYMBOLICA_COMMUNITY_PATH' in this standalone Python runner.",]))

from symbolica_community.tensors import TensorNetwork, Representation, TensorStructure, TensorIndices, Tensor, Slot  # type: ignore # nopep8
from symbolica_community import CompiledEvaluator, S, E  # type: ignore # nopep8

N_EXTERNALS = %(n_externals)s
MODEL_PARAMS = [
    %(model_parameters)s
]

N_HORNER_ITERATIONS = 100
N_CORES = 1
OPRIMIZATION_LEVEL = 3
COMPILER = None  # '/opt/local/bin/gcc'
N_SAMPLES = 2
N_REPEATED_SAMPLES = 1_000_000
SEED = 1337

if __name__ == "__main__":
    print("Running standalone script")

    with open("me_expression.txt", "r") as f:
        me_expression = E(f.read())

    n_external = 4
    params = []
    for i in range(N_EXTERNALS):
        for idx in range(4):
            params.append(E(f"kin_p{i+1}_{idx}"))

    model_params = [mp[0] for mp in MODEL_PARAMS]
    model_params_values = [mp[1] for mp in MODEL_PARAMS]

    if not os.path.isfile("ME.so"):
        me_evaluator = me_expression.evaluator(constants={}, functions={}, params=params+model_params, iterations=N_HORNER_ITERATIONS, n_cores=N_CORES, verbose=True)  # nopep8
        compiled_evaluator = me_evaluator.compile(
            function_name="ME", filename="ME", library_name="ME.so", inline_asm="none", optimization_level=OPRIMIZATION_LEVEL, compiler_path=None)
    else:
        compiled_evaluator = CompiledEvaluator.load(
            filename="ME.so", function_name="ME", input_len=len(params)+len(model_params), output_len=1)

    ps_gen = PSGen(initial_masses=[0.,]*2, final_masses=[0.,]*(N_EXTERNALS-2),
                   beam_Es=[500., 500.],
                   beam_types=(0, 0))

    print("Generating samples ...")
    random.seed(SEED)
    samples = []
    for i_s in range(N_SAMPLES):
        rv = [random.random() for _ in range(ps_gen.nDimPhaseSpace())]
        kinematics, _jac = ps_gen.generateKinematics(E_cm=1000., random_variables=rv)  # nopep8
        print("> Sample #{}:\n{}".format(i_s, kinematics))
        param_inputs = []
        for i in range(N_EXTERNALS):
            for idx in range(4):
                param_inputs.append(kinematics[i][idx])

        param_inputs += model_params_values
        samples.append(param_inputs)

    samples.extend([samples[-1],]*N_REPEATED_SAMPLES)
    print("Starting evaluations ...")
    t_start = time.time()
    results = compiled_evaluator.evaluate_complex(samples)
    t_tot = time.time() - t_start
    for i_s in range(N_SAMPLES):
        print("> Result for sample #{}: {}".format(i_s, results[i_s][0]))
    print("Time per sample: {:.3f} mus".format(
          ((t_tot / (N_SAMPLES + N_REPEATED_SAMPLES)) * 1_000_000)))
