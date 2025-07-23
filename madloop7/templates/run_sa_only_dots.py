from phase_space_generator import FlatInvertiblePhasespace as PSGen
from phase_space_generator import LorentzVector, LorentzVectorList
import sys
import os
import random
import time
import multiprocessing

pjoin = os.path.join

RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"

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

N_HORNER_ITERATIONS = 5
N_CORES = multiprocessing.cpu_count()
OPRIMIZATION_LEVEL = %(optimisation_level)d
INLINE_ASM = %(inline_asm)s
COMPILER = None  # '/opt/local/bin/gcc'
N_SAMPLES = 2
N_REPEATED_SAMPLES = 1_000
SEED = 1337
TARGETS = %(targets)s
EXPAND_BEFORE_BUILDING_EVALUATOR = %(expand_before_building_evaluator)s
THRESHOLD = 1.0e-10  # Threshold for comparison to benchmark

if __name__ == "__main__":
    print("Running standalone script")

    with open("TEST_OUTCOME.txt", "w") as f:
        f.write("FAIL")

    with open("me_expression_terms.txt", "r") as f:
        me_expression_terms = [E(t) for t in eval(f.read())]
    me_expression = E("0")
    for t in me_expression_terms:
        me_expression += t

    n_external = 4
    params = []
    for i in range(N_EXTERNALS):
        for j in range(N_EXTERNALS):
            params.append(E(f"dot_n{i+1}_{j+1}"))
            if i <= j:
                params.append(E(f"dot_{i+1}_{j+1}"))
                params.append(E(f"dot_n{i+1}_n{j+1}"))

    model_params = [mp[0] for mp in MODEL_PARAMS]
    model_params_values = [mp[1] for mp in MODEL_PARAMS]

    if EXPAND_BEFORE_BUILDING_EVALUATOR:
        print("Expanding ME expression of size {} bytes before building evaluator ...".format(me_expression.get_byte_size()))  # nopep8
        me_expression = me_expression.expand()
        print("ME expression after expansion: size {} bytes".format(me_expression.get_byte_size()))  # nopep8
    if not os.path.isfile("ME.so"):
        me_evaluator = me_expression.evaluator(constants={}, functions={}, params=params+model_params, iterations=N_HORNER_ITERATIONS, n_cores=N_CORES, verbose=True)  # nopep8
        compiled_evaluator = me_evaluator.compile(
            function_name="ME", filename="ME", library_name="ME.so", inline_asm=INLINE_ASM, optimization_level=OPRIMIZATION_LEVEL, compiler_path=None)
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
        kinematics_n = LorentzVectorList([LorentzVector([
            lv[0], -lv[1], -lv[2], -lv[3]]) for lv in kinematics])
        param_inputs = []
        for i in range(N_EXTERNALS):
            for j in range(N_EXTERNALS):
                param_inputs.append(kinematics_n[i].dot(kinematics[j]))
                if i <= j:
                    param_inputs.append(kinematics[i].dot(kinematics[j]))
                    param_inputs.append(kinematics_n[i].dot(kinematics_n[j]))
        param_inputs += model_params_values
        samples.append(param_inputs)

    samples.extend([samples[-1],]*N_REPEATED_SAMPLES)
    print("Starting evaluations ...")
    t_start = time.time()
    results = compiled_evaluator.evaluate_complex(samples)
    t_tot = time.time() - t_start

    with open("TEST_OUTCOME.txt", "w") as f:
        f.write("SUCCESS")

    for i_s in range(N_SAMPLES):
        print("> Result for sample #{}       : {}".format(
            i_s, results[i_s][0]))
        if i_s < len(TARGETS):
            print("> Target result for sample #{}: {}".format(
                i_s, TARGETS[i_s]))
            diff = abs(abs(results[i_s][0]) - abs(TARGETS[i_s]))/(abs(abs(results[i_s][0]) + abs(TARGETS[i_s])))  # nopep8
            if diff > THRESHOLD:
                print("{}> Relative difference: {}{}".format(RED, diff, RESET))
                with open("TEST_OUTCOME.txt", "w") as f:
                    f.write("FAIL")
            else:
                print("{}> Relative difference: {}{}".format(GREEN, diff, RESET))
            print("> Relative difference: {}".format(diff))

    print("Time per sample: {:.3f} mus".format(
          ((t_tot / (N_SAMPLES + N_REPEATED_SAMPLES)) * 1_000_000)))
