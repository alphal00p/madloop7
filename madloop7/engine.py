from . import logger
from pprint import pformat
from madloop7.process_definitions import HardCodedProcess, HARDCODED_PROCESSES
from madloop7.utils import MadLoop7Error, get_model, FlatPhaseSpaceGenerator
from typing import Any

import yaml  # type: ignore
import json
import os
import sys
import logging
import subprocess
import shutil
from enum import Enum

from pprint import pformat, pprint

root_path = os.path.abspath(os.path.dirname(__file__))

pjoin = os.path.join

graphs_output_DIR = "graphs_outputs"
MADGRAPH_OUTPUT_DIR = "madgraph_outputs"
GAMMALOOP_OUTPUT_DIR = "gammaloop_outputs"
MADLOOP7_OUTPUT_DIR = "madloop7_outputs"

PHYSICAL_VECTOR_SUM_RULE = True


class EvalMode(Enum):
    """Evaluation modes for MadLoop7."""
    TREExTREE = "tree-level"
    LOOPxTREE = "virtual"
    LOOPxLOOP = "loop-induced"


class EvaluationStrategy(Enum):
    """Evaluation strategies for MadLoop7."""
    ONLY_DOT_PRODUCTS = "only_dot_products"
    TENSOR_NETWORKS = "tensor_networks"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, EvaluationStrategy):
            return self.value == other.value
        elif isinstance(other, str):
            return self.value == other
        return False

    @classmethod
    def from_string(cls, value: str) -> "EvaluationStrategy":
        match value:
            case "only_dot_products":
                return EvaluationStrategy.ONLY_DOT_PRODUCTS
            case "tensor_networks":
                return EvaluationStrategy.TENSOR_NETWORKS
            case _:
                raise ValueError(f"Invalid evaluation strategy: {value}")


class MadLoop7(object):

    def __init__(self, config_path: str) -> None:
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def generate_process_gamma_loop(self, process: HardCodedProcess) -> None:
        assert process.gamma_loop_generation is not None
        assert process.graphs_output is not None
        logger.info(f"Generating process {process.name} with GammaLoop (feyngen)...")  # nopep8
        process_card = os.path.abspath(pjoin(self.config["output_path"], GAMMALOOP_OUTPUT_DIR, f"{process.name}.gL"))  # nopep8
        graph_outputs = [
            (
                graph_class,
                generation_cmds,
                os.path.abspath(pjoin(self.config["output_path"], GAMMALOOP_OUTPUT_DIR, f"{process.name}_{graph_class}.dot"))  # nopep8
             )  # nopep8
            for (generation_cmds, (graph_class, _)) in zip(process.gamma_loop_generation, process.graphs_output)
        ]
        with open(process_card, "w") as f:
            logger.debug(f"Writing gammaloop process generation card to '{process_card}'")  # nopep8
            f.write("\n".join(
                [
                    f"import_model {process.model}",
                ]
                + ['\n',]
                + [
                    "\n".join([
                        "\n".join(l.strip()+(' --clear_existing_processes --graph_prefix GL_' if i == len(generation_cmds.strip().split(
                            '\n'))-1 else '') for i, l in enumerate(generation_cmds.strip().split('\n'))),
                        f"export_graphs {dot_output}",  # nopep8
                    ])
                    for (_graph_class, generation_cmds, dot_output) in graph_outputs
                ]
            ))

        logger.debug(f"Running gammaloop graph generation...")
        process_handle = subprocess.Popen(
            ["./bin/gammaloop", process_card],
            stdout=None if logger.level <= logging.DEBUG else subprocess.PIPE,
            stderr=None if logger.level <= logging.DEBUG else subprocess.PIPE,
            cwd=self.config["gammaloop_path"]
        )

        if logger.level > logging.DEBUG:
            stdout, stderr = process_handle.communicate()
            gl_output = f"stdout:\n{stdout.decode()}\nstderr:\n{stderr.decode()}"  # nopep8
        else:
            process_handle.wait()
            gl_output = "<DEBUG MODE: see terminal>"

        for (_graph_class, generation_cmds, dot_output) in graph_outputs:
            if not os.path.isfile(dot_output):
                raise MadLoop7Error(
                    f"GammaLoop generation failed for process {process.name}:\n{gl_output}")

        logger.info(f"Process {process.name} generated.")
        logger.debug(f"GammaLoop output:\n{gl_output}")

    def generate_process_madgraph(self, process: HardCodedProcess) -> None:
        assert process.madgraph_generation is not None
        assert process.graphs_output is not None
        logger.info(f"Generating process {process.name} with MG5aMC...")  # nopep8
        process_card = os.path.abspath(pjoin(self.config["output_path"], MADGRAPH_OUTPUT_DIR, f"{process.name}.mg5"))  # nopep8
        yaml_graph_outputs = os.path.abspath(pjoin(self.config["output_path"], graphs_output_DIR, f"{process.name}"))  # nopep8
        madgraph_output = os.path.abspath(pjoin(self.config["output_path"], MADGRAPH_OUTPUT_DIR, process.name))  # nopep8
        graph_outputs = [
            (
                graph_class,
                os.path.abspath(pjoin(self.config["output_path"], graphs_output_DIR, process.name, ms_output)),  # nopep8
                os.path.abspath(pjoin(self.config["output_path"], GAMMALOOP_OUTPUT_DIR, f"{process.name}_{graph_class}.dot"))  # nopep8
             )  # nopep8
            for (graph_class, ms_output) in process.graphs_output
        ]
        with open(process_card, "w") as f:
            logger.debug(f"Writing MG5aMC process generation card to '{process_card}'")  # nopep8
            f.write("\n".join(
                [
                    f"set_madsymbolic_option gammaloop_path {self.config['gammaloop_path']}",  # nopep8
                    f"import model {process.model}",
                ]
                + [process.madgraph_generation,]
                + [
                    f"write_graphs {yaml_graph_outputs} --format yaml",
                    f"output standalone {madgraph_output}",
                ]
                + [
                    "\n".join([
                        f"gL import_graphs {graphs_output} --format yaml",  # nopep8
                        f"gL export_graphs {dot_output}",  # nopep8
                    ])
                    for (_graph_class, graphs_output, dot_output) in graph_outputs
                ]
            ))
        logger.debug(f"Running MG5aMC generation...")
        process_handle = subprocess.Popen(
            ["./bin/mg5_aMC", "--mode", "madsymbolic", process_card],
            stdout=None if logger.level <= logging.DEBUG else subprocess.PIPE,
            stderr=None if logger.level <= logging.DEBUG else subprocess.PIPE,
            cwd=self.config["madgraph_path"]
        )

        if logger.level > logging.DEBUG:
            stdout, stderr = process_handle.communicate()
            mg_output = f"stdout:\n{stdout.decode()}\nstderr:\n{stderr.decode()}"  # nopep8
        else:
            process_handle.wait()
            mg_output = "<DEBUG MODE: see terminal>"

        for (_graph_class, _graphs_output, dot_output) in graph_outputs:
            if not os.path.isfile(dot_output):
                raise MadLoop7Error(
                    f"MadGraph generation failed for process {process.name}:\n{mg_output}")

        logger.info(f"Process {process.name} generated.")
        logger.debug(f"MG5aMC output:\n{mg_output}")

    def generate_process(self, process: HardCodedProcess) -> None:
        if process.gamma_loop_generation is not None:
            self.generate_process_gamma_loop(process)
        elif process.madgraph_generation is not None:
            self.generate_process_madgraph(process)
        else:
            raise MadLoop7Error(f"Generation logic for process {process.name} not supported")  # nopep8

    def build_expressions_with_gammaloop(self, process: HardCodedProcess) -> None:
        logger.info("Build diagram expressions with GammaLoop...")
        self.import_gammaloop()
        from gammaloop.interface.gammaloop_interface import GammaLoop, CommandList  # type: ignore # nopep8

        for graph_class in process.get_graph_categories():
            graphs = os.path.abspath(
                pjoin(self.config["output_path"], GAMMALOOP_OUTPUT_DIR, f"{process.name}_{graph_class}.dot"))
            gl_output = os.path.abspath(
                pjoin(self.config["output_path"], GAMMALOOP_OUTPUT_DIR, f"{process.name}_{graph_class}"))

            gL_runner = GammaLoop()
            gL_runner.run(CommandList.from_string(
                f"""
                import_model {process.model} --format ufo
                import_graphs {graphs} --format dot
                output {gl_output} --expression_format file -mr -exp --no_evaluators --yaml_only
                """
            ))

            if not os.path.isdir(gl_output):
                raise MadLoop7Error(
                    f"GammaLoop failed for process {process.name}")

    def build_evaluators_with_spenso(self, process: HardCodedProcess, tree_graph_ids: list[int] | None = None, loop_graph_ids: list[int] | None = None, evaluation_strategy: EvaluationStrategy | None = None) -> None:

        logger.info("Build graph evaluators with spenso and strategy '%s'...", str(evaluation_strategy))  # nopep8

        self.import_symbolica_community()
        from symbolica_community import E  # type: ignore # nopep8

        def evaluate_graph_overall_factor(overall_factor: E) -> E:
            for header in ["AutG",
                           "CouplingsMultiplicity",
                           "InternalFermionLoopSign",
                           "ExternalFermionOrderingSign",
                           "AntiFermionSpinSumSign",
                           "NumeratorIndependentSymmetryGrouping"]:
                overall_factor = overall_factor.replace(
                    E(f"{header}(x_)"), E("x_"), repeat=True)
            overall_factor = overall_factor.replace(
                E("NumeratorDependentGrouping(GraphId_,ratio_,GraphSymmetryFactor_)"), E("ratio_*GraphSymmetryFactor_"), repeat=True)
            return overall_factor.expand().collect_num()

        if evaluation_strategy is None:
            evaluation_strategy = EvaluationStrategy.TENSOR_NETWORKS

        expressions: dict[Any, Any] = {}
        for graph_class in ['tree', 'loop']:
            expressions[graph_class] = {
                'model_replacements': {},
                'graph_expressions': {},
            }
        for graph_class in process.get_graph_categories():
            output_name = f"{process.name}_{graph_class}"
            # Load model replacements
            with open(pjoin(self.config["output_path"], GAMMALOOP_OUTPUT_DIR, output_name, "output_metadata.yaml"), "r") as f:
                output_metadata = yaml.safe_load(f)

            with open(pjoin(self.config["output_path"], GAMMALOOP_OUTPUT_DIR, output_name, "sources", "model", "model_replacements.json"), "r") as f:
                model_replacements = dict(json.load(f))

            graph_expressions = {}
            for graph_expr_path in os.listdir(pjoin(self.config["output_path"], GAMMALOOP_OUTPUT_DIR, output_name, "sources", output_metadata["output_type"], output_metadata["contents"][0], "expressions")):
                print(graph_expr_path)
                graph_id = int(graph_expr_path.split("_")[-2])
                graph_ids = tree_graph_ids if graph_class == "tree" else loop_graph_ids
                if graph_ids is None or graph_id in graph_ids:
                    with open(pjoin(self.config["output_path"], GAMMALOOP_OUTPUT_DIR, output_name, "sources", output_metadata["output_type"], output_metadata["contents"][0], "expressions", graph_expr_path), "r") as f:
                        expr = json.load(f)
                        graph_expressions[graph_id] = {
                            'expression': expr[0],
                            'overall_factor': evaluate_graph_overall_factor(E(expr[1])),
                            'momenta': dict(expr[2]),
                            'denominators': dict(expr[3]),
                        }

            expressions[graph_class] = {
                'model_replacements': model_replacements,
                'graph_expressions': graph_expressions,
            }

        # logger.debug("Model replacements:\n{}".format(pformat(model_replacements)))

        eval_mode = EvalMode.TREExTREE
        if expressions["loop"]["graph_expressions"] == {} and expressions["tree"]["graph_expressions"] == {}:
            raise MadLoop7Error(f"No graphs to interefere for {process.name}")

        if expressions["loop"]["graph_expressions"] != {} and expressions["tree"]["graph_expressions"] != {}:
            eval_mode = EvalMode.LOOPxTREE
        elif expressions["tree"]["graph_expressions"] == {}:
            eval_mode = EvalMode.LOOPxLOOP

        # self.import_symbolica_community()
        # from symbolica_community import Expression, S, E  # type: ignore # nopep8
        # print((S("TTT", is_linear=True, is_symmetric=True)(
        #     E("x+y"), E("x+y"))**-1).expand().to_canonical_string())
        # stop
        match evaluation_strategy:
            case EvaluationStrategy.ONLY_DOT_PRODUCTS:
                logger.info("Building evaluator with only dot products...")
                matrix_element_expression = self.build_matrix_element_expression_only_dots(
                    process, expressions,  eval_mode)
                self.export_evaluator_only_dots(
                    process, matrix_element_expression, eval_mode)
            case EvaluationStrategy.TENSOR_NETWORKS:
                logger.info("Building evaluator with tensor networks...")
                matrix_element_expression = self.build_matrix_element_expression_tensor_networks(
                    process, expressions,  eval_mode)
                self.export_evaluator_tensor_networks(
                    process, matrix_element_expression, eval_mode)

    def build_graph_expression(self, graph_expr: dict[Any, Any], to_lmb: bool = False, split_num_denom: bool = False) -> Any:
        """Build the graph expression from the graph expression dictionary."""
        self.import_symbolica_community()
        from symbolica_community import Expression, S, E  # type: ignore # nopep8

        def E_sp(expr: str) -> Expression:
            """Create a symbolica_community expression with the spenso namespace."""
            return E(expr, default_namespace="spenso")

        def curate(expr: Expression) -> Expression:
            expr = expr.replace(E_sp("Metric(x_,y_)"),
                                E_sp("g(x_,y_)"), repeat=True)
            expr = expr.replace(E_sp("id(x_,y_)"),
                                E_sp("𝟙(x_,y_)"), repeat=True)
            expr = expr.replace(E("spenso::γ(x_,y_,z_)"), E(
                "alg::gamma(x_,y_,z_)"), repeat=True)
            expr = expr.replace(E("spenso::T(x_,y_,z_)"),
                                E("alg::t(x_,y_,z_)"), repeat=True)
            expr = expr.replace(E("spenso::f(x_,y_,z_)"),
                                E("alg::f(x_,y_,z_)"), repeat=True)
            expr = expr.replace(E("spenso::TR"), E("alg::TR"), repeat=True)
            expr = expr.replace(E("spenso::Nc"), E("alg::Nc"), repeat=True)
            expr = expr.replace(E("spenso::v(x__)"),
                                E("alg::v(x__)"), repeat=True)
            expr = expr.replace(E("spenso::vbar(x__)"),
                                E("alg::vbar(x__)"), repeat=True)
            expr = expr.replace(E("spenso::u(x__)"),
                                E("alg::u(x__)"), repeat=True)
            expr = expr.replace(E("spenso::ubar(x__)"),
                                E("alg::ubar(x__)"), repeat=True)
            expr = expr.replace(E("spenso::ϵ(x__)"),
                                E("alg::ϵ(x__)"), repeat=True)
            expr = expr.replace(E("spenso::ϵbar(x__)"),
                                E("alg::ϵbar(x__)"), repeat=True)
            expr = expr.replace(E("spenso::mink(4,x_)"), E(
                "spenso::mink(python::dim,x_)"), repeat=True)
            expr = expr.replace(E("spenso::coad(8,x_)"), E(
                "spenso::coad(alg::Nc^2-1,x_)"), repeat=True)
            expr = expr.replace(E("spenso::cof(3,x_)"), E(
                "spenso::cof(alg::Nc,x_)"), repeat=True)
            return expr

        numerator = curate(
            E_sp(graph_expr['expression']))*graph_expr['overall_factor']
        denominator = E("1")
        for (p, m) in graph_expr['denominators'].items():
            denominator = denominator * \
                (S("symbolica_community::dot", is_linear=True,
                 is_symmetric=True)(E_sp(p), E_sp(p)) - E_sp(m)**2)

        if to_lmb:
            for (src, trgt) in graph_expr['momenta'].items():
                numerator = numerator.replace(
                    E_sp(src), E_sp(trgt), repeat=True)
                denominator = denominator.replace(
                    E_sp(src), E_sp(trgt), repeat=True)

        if split_num_denom:
            return (numerator, denominator)
        else:
            return numerator / denominator

    def sum_square_left_right_only_dots(self, left: dict[Any, Any], right: dict[Any, Any]) -> Any:
        """Combine left and right expressions into a single expression."""
        self.import_symbolica_community()
        from symbolica_community import Expression, S, E  # type: ignore # nopep8
        from symbolica_community.tensors import TensorNetwork, Representation, TensorStructure, TensorIndices, Tensor, Slot  # type: ignore # nopep8
        from symbolica_community.algebraic_simplification import wrap_dummies, conj, simplify_color, simplify_gamma, to_dots  # type: ignore # nopep8
        expressions = {'left': left, 'right': right}

        dim = S("python::dim")
        mink = Representation.mink(dim)
        P_symbol = S("spenso::P")
        N_symbol = S("spenso::N")
        Q = TensorStructure(mink, name=S("spenso::Q"))
        P = TensorStructure(mink, name=P_symbol)
        N = TensorStructure(mink, name=N_symbol)

        def q(i, j):
            return Q(i, ';', j)

        def p(i, j):
            return P(i, ';', j)

        def n(i, j):
            return N(i, ';', j)

        gamma = TensorStructure.gammadD(dim)
        g = TensorStructure.metric(mink)

        bis = Representation.bis(4)
        u, ubar, v, vbar, eps, epsbar = S(
            "alg::u", "alg::ubar", "alg::v", "alg::vbar", "alg::ϵ", "alg::ϵbar")
        i_, j_, d_, a_, b_ = S("i_", "j_", "d_", "a_", "b_")
        dummy = S("dummy")
        l_side = S("l")
        r_side = S("r")
        dot = S("symbolica_community::dot", is_linear=True, is_symmetric=True)

        SIMPLIFY_INDIVIDUAL_INTERFERENCE_TERMS_SEPARATELY = True

        terms = []
        if not SIMPLIFY_INDIVIDUAL_INTERFERENCE_TERMS_SEPARATELY:
            for key in ['left', 'right']:
                side_expression = E("0")
                for _graph_id, graph_expr in sorted(expressions[key].items(), key=lambda x: x[0]):
                    side_expression = side_expression + \
                        self.build_graph_expression(graph_expr, to_lmb=True)
                expressions[key] = side_expression

            left_e = wrap_dummies(expressions['left'], l_side)
            right_e = conj(wrap_dummies(expressions['right'], r_side))
            terms.append((left_e*right_e).expand())
        else:
            diagram_expressions = {}
            diagram_expressions['left'] = [
                (graph_id, wrap_dummies(self.build_graph_expression(
                    graph_expr, to_lmb=True), l_side))
                for graph_id, graph_expr in sorted(expressions['left'].items(), key=lambda x: x[0])
            ]
            diagram_expressions['right'] = [
                (graph_id, conj(wrap_dummies(self.build_graph_expression(
                    graph_expr, to_lmb=True), r_side)))
                for graph_id, graph_expr in sorted(expressions['right'].items(), key=lambda x: x[0])
            ]
            for left_id, left_expr in diagram_expressions['left']:
                for right_id, right_expr in diagram_expressions['right']:
                    terms.append((left_expr*right_expr).expand())

        # For now only support massless externals
        if PHYSICAL_VECTOR_SUM_RULE:
            transverse_physical_vector_sum = (
                p(i_, l_side(a_)) * n(i_, r_side(a_)) +
                n(i_, l_side(a_)) * p(i_, r_side(a_))
            ) / dot(N_symbol(i_), P_symbol(i_))
        else:
            transverse_physical_vector_sum = E("0")

        spin_sum_rules = [
            (
                eps(i_, mink(l_side(a_)))*epsbar(i_, mink(r_side(a_))),
                -g(l_side(a_), r_side(a_)) + transverse_physical_vector_sum
            ),
            (
                eps(i_, mink(r_side(a_)))*epsbar(i_, mink(l_side(a_))),
                -g(l_side(a_), r_side(a_)) + transverse_physical_vector_sum
            ),
            (
                vbar(i_, bis(l_side(a_)))*v(i_, bis(r_side(a_))),
                gamma(dummy(i_, a_), l_side(a_), r_side(a_)) *
                p(i_, dummy(i_, a_))
            ),
            (
                vbar(i_, bis(r_side(a_)))*v(i_, bis(l_side(a_))),
                gamma(dummy(i_, a_), r_side(a_), l_side(a_)) *
                p(i_, dummy(i_, a_))
            ),
            (
                ubar(i_, bis(l_side(a_)))*u(j_, bis(r_side(a_))),
                -gamma(dummy(i_, a_), l_side(a_), r_side(a_)) *
                p(i_, dummy(i_, a_))
            ),
            (
                ubar(i_, bis(r_side(a_)))*u(j_, bis(l_side(a_))),
                -gamma(dummy(i_, a_), r_side(a_), l_side(a_)) *
                p(i_, dummy(i_, a_))
            ),
        ]

        def substitute_constants(input_e: Expression) -> Expression:
            constants = [
                (dim, E("4")),
                (E("alg::TR"), E("1/2")),
                (E("alg::Nc"), E("3")),
                (E("alg::CF"), E("4/3")),
                (E("alg::CA"), E("3")),
            ]
            for src, trgt in constants:
                input_e = input_e.replace(src, trgt)
            return input_e

        final_result = E("0")
        logger.info(
            "Starting the processing of %d interference terms..." % len(terms))
        for i_t, t in enumerate(terms):
            print("Processing terms %-6d / %d " %
                  (i_t, len(terms)), end="\r")
            for src, trgt in spin_sum_rules:
                t = t.replace(src, trgt, repeat=True)
            # with open('/Users/vjhirsch/Documents/Work/madloop7/test_expression_before_color_simplify.txt', 'w') as f:
            #     f.write(t.to_canonical_string())
            # print("Simplifying_color...")
            t = simplify_color(t)
            # with open('/Users/vjhirsch/Documents/Work/madloop7/test_expression_before_gamma_simplify.txt', 'w') as f:
            #     f.write(t.to_canonical_string())
            # print("Simplifying_gamma...")
            t = simplify_gamma(t)
            t = to_dots(t)
            t = substitute_constants(t)
            final_result = final_result + t

        return final_result

    def build_matrix_element_expression_only_dots(self, process: HardCodedProcess, expressions: dict[Any, Any], mode: EvalMode) -> Any:

        self.import_symbolica_community()
        from symbolica_community import Expression, S, E  # type: ignore # nopep8

        match mode:
            case EvalMode.TREExTREE:
                me_expr = self.sum_square_left_right_only_dots(
                    expressions["tree"]["graph_expressions"], expressions["tree"]["graph_expressions"])
                me_expr = me_expr * E(process.overall_factor)
                return me_expr

            case EvalMode.LOOPxTREE:
                raise MadLoop7Error(
                    "Loop x tree matrix element expression building not implemented yet!")
            case EvalMode.LOOPxLOOP:
                raise MadLoop7Error(
                    "Loop x loop matrix element expression building not implemented yet!")

    def export_evaluator_only_dots(self, process: HardCodedProcess, matrix_element_expression: Any, mode: EvalMode) -> None:
        self.import_symbolica_community()
        from symbolica_community.tensors import TensorNetwork, Representation, TensorStructure, TensorIndices, Tensor, Slot  # type: ignore # nopep8
        from symbolica_community import Expression, S, E  # type: ignore # nopep8

        madloop7_output = os.path.abspath(pjoin(self.config["output_path"], MADLOOP7_OUTPUT_DIR, process.name))  # nopep8
        if not os.path.isdir(madloop7_output):
            os.makedirs(madloop7_output)

        dim = S("python::dim")
        mink = Representation.mink(dim)
        P = S("spenso::P")
        N = S("spenso::N")
        dot = S("symbolica_community::dot", is_linear=True, is_symmetric=True)
        match mode:
            case EvalMode.TREExTREE:

                # Overall energy-momentum conservation
                # complement = E("0")
                # complement_n = E("0")
                # for i in range(process.n_external-1):
                #     if i < 2:
                #         complement = complement + P(i)
                #         complement_n = complement_n + N(i)
                #     else:
                #         complement = complement - P(i)
                #         complement_n = complement_n - N(i)
                # matrix_element_expression = matrix_element_expression.replace(
                #     P(process.n_external-1), complement)
                # matrix_element_expression = matrix_element_expression.replace(
                #     N(process.n_external-1), complement_n)

                for i in range(process.n_external):
                    for j in range(process.n_external):
                        matrix_element_expression = matrix_element_expression.replace(
                            dot(N(i), P(j)), E(f"dot_n{i+1}_{j+1}"))
                        if i <= j:
                            if i == j:
                                # Only support massless
                                dot_param = E("0")
                            else:
                                dot_param = E(f"dot_{i+1}_{j+1}")
                            matrix_element_expression = matrix_element_expression.replace(
                                dot(P(i), P(j)), dot_param)
                            matrix_element_expression = matrix_element_expression.replace(
                                dot(N(i), N(j)), E(f"dot_n{i+1}_n{j+1}"))

                model_param_symbols = [s for s in matrix_element_expression.get_all_symbols(
                    False) if not str(s).startswith("dot")]
                model = get_model(process.model)
                model_parameters = model.get('parameter_dict')
                matrix_element_expression = matrix_element_expression.expand().collect_factors()

                with open(pjoin(madloop7_output, "me_expression.txt"), "w") as f:
                    f.write(matrix_element_expression.to_canonical_string())
                shutil.copyfile(
                    pjoin(root_path, "templates", "phase_space_generator.py"),
                    pjoin(madloop7_output, "phase_space_generator.py"),
                )
                run_sa_template = open(
                    pjoin(root_path, "templates", "run_sa_only_dots.py"), 'r').read()
                replace_dict = {
                    'symbolica_community_path': self.config["symbolica_community_path"],
                    'n_externals': process.n_external,
                }
                model_param_values = []
                for p in model_param_symbols:
                    p_name = str(p).replace('spenso::', '')
                    if p_name not in model_parameters:
                        p_name = f'mdl_{p_name}'
                    model_param_values.append(
                        (p, model_parameters[p_name].value)
                    )
                replace_dict['model_parameters'] = ',\n'.join(
                    [f"(S(\"{p.to_canonical_string()}\"), {v})" for (p, v) in model_param_values])

                with open(pjoin(madloop7_output, "run.py"), "w") as f:
                    f.write(run_sa_template % replace_dict)

            case EvalMode.LOOPxTREE:
                raise MadLoop7Error(
                    "Loop x tree matrix element expression evaluation not implemented yet!")
            case EvalMode.LOOPxLOOP:
                raise MadLoop7Error(
                    "Loop x loop matrix element expression evaluation not implemented yet!")

    def expand_color(self, expression) -> list[tuple[Any, Any]]:
        self.import_symbolica_community()
        from symbolica_community import Expression, S, E, Replacement  # type: ignore # nopep8

        f_, x_, x___, y__, z___ = S("f_", "x_", "x___", "y__", "z___")
        color, coad, cof, dind = S(
            "color", "spenso::coad", "spenso::cof", "spenso::dind")
        tmp = expression.replace_multiple([
            Replacement(f_(x___, coad(y__), z___),
                        color(f_(x___, coad(y__), z___))),
            Replacement(f_(x___, cof(y__), z___),
                        color(f_(x___, cof(y__), z___))),
            Replacement(f_(x___, dind(cof(y__)), z___),
                        color(f_(x___, dind(cof(y__)), z___))),
        ])
        color_structures = [m[x_] for m in tmp.match(color(x_))]
        tmp = tmp.replace(color(x_), x_)

        return tmp.coefficient_list(*color_structures)

    def expand_pol_vectors(self, expression) -> list[tuple[Any, Any]]:
        self.import_symbolica_community()
        from symbolica_community import Expression, S, E, Replacement  # type: ignore # nopep8

        x_, x___ = S("x_", "x___")
        polarization_vec = S("polarization_vec")
        u, ubar, v, vbar, eps, epsbar = S(
            "alg::u", "alg::ubar", "alg::v", "alg::vbar", "alg::ϵ", "alg::ϵbar")

        tmp = expression.replace_multiple([
            Replacement(u(x___), polarization_vec(u(x___))),
            Replacement(ubar(x___), polarization_vec(ubar(x___))),
            Replacement(v(x___), polarization_vec(v(x___))),
            Replacement(vbar(x___), polarization_vec(vbar(x___))),
            Replacement(eps(x___), polarization_vec(eps(x___))),
            Replacement(epsbar(x___), polarization_vec(epsbar(x___))),
        ])
        polarization_vec_structures = [m[x_]
                                       for m in tmp.match(polarization_vec(x_))]
        tmp = tmp.replace(polarization_vec(x_), x_)

        return tmp.coefficient_list(*polarization_vec_structures)

    def sum_square_left_right_tensor_networks(self, process: HardCodedProcess, left: dict[Any, Any], right: dict[Any, Any]) -> Any:
        """Combine left and right expressions into a single expression."""
        self.import_symbolica_community()
        from symbolica_community import Expression, S, E  # type: ignore # nopep8
        from symbolica_community.tensors import sparse_empty, dense, TensorLibrary, TensorNetwork, Representation, TensorStructure, TensorIndices, Tensor, Slot  # type: ignore # nopep8
        from symbolica_community.algebraic_simplification import cook_indices, wrap_dummies, conj, simplify_color, simplify_gamma, to_dots  # type: ignore # nopep8

        dim = S("python::dim")
        mink = Representation.mink(dim)
        mink4 = Representation.mink(4)
        Q_symbol = S("spenso::Q")
        P_symbol = S("spenso::P")
        N_symbol = S("spenso::N")
        Q = TensorStructure(mink, name=Q_symbol)
        P = TensorStructure(mink, name=P_symbol)
        N = TensorStructure(mink, name=N_symbol)
        Q4 = TensorStructure(mink4, name=Q_symbol)
        P4 = TensorStructure(mink4, name=P_symbol)
        N4 = TensorStructure(mink4, name=N_symbol)

        def q(i, j):
            return Q(i, ';', j)

        def p(i, j):
            return P(i, ';', j)

        def n(i, j):
            return N(i, ';', j)

        def q4(i, j):
            return Q4(i, ';', j)

        def p4(i, j):
            return P4(i, ';', j)

        def n4(i, j):
            return N4(i, ';', j)

        my_lib = TensorLibrary.weyl()

        g_struct = TensorStructure.metric(mink4)

        metric_tensor = sparse_empty(g_struct, type(1.0))
        metric_tensor[[0, 0]] = 1
        metric_tensor[[1, 1]] = -1
        metric_tensor[[2, 2]] = -1
        metric_tensor[[3, 3]] = -1

        my_lib.register(metric_tensor)
        id_struct = TensorStructure.id(mink4)

        # my_tensor = TensorNetwork(P4(2)*TensorStructure.id(mink4)(2, 3))
        # my_tensor = TensorNetwork(p4(69, 2)*TensorStructure.id(mink4)(2, 3))
        # my_tensor.execute(my_lib)
        # print(my_tensor.result_tensor(my_lib))
        # print(my_tensor.result_tensor(my_lib)[1])

        # # my_tensor = TensorNetwork(P4(2)*TensorStructure.id(mink4)(2, 3))
        # # my_tensor = TensorNetwork.one()*P4(2)

        # my_tensor = TensorNetwork.from_expression(
        #     P4(2)*id_struct(2, 3), my_lib)
        # my_tensor = TensorNetwork.from_expression(P4(2)*g_struct(2, 3), my_lib)

        # my_tensor = TensorNetwork.from_expression(
        #     cook_indices(E("spenso::g(spenso::mink(4,l(2)),spenso::mink(4,r(2)))*P(2,spenso::mink(4,l(2)))")), my_lib)
        # my_tensor = TensorNetwork.from_expression(
        #     cook_indices(E("spenso::g(spenso::mink(4,l(11)),spenso::mink(4,r(22)))*spenso::𝟙(spenso::mink(4,l(2)),spenso::mink(4,r(2)))*P(2,spenso::mink(4,l(2)))")), my_lib)
        # # my_tensor = TensorNetwork.from_expression(
        # #     cook_indices(E("P(2,spenso::mink(4,l(2)))")), my_lib)

        # my_tensor.execute(my_lib)
        # print(my_tensor.result_tensor(my_lib))
        # print(my_tensor.result_tensor(my_lib).structure())

        l_side = S("l")
        r_side = S("r")
        # pol_sums = []
        # for i in range(process.n_external):
        #     pol_sum_i = TensorNetwork.from_expression(
        #         cook_indices(
        #             - E(f"spenso::g(spenso::mink(4,l({i})),spenso::mink(4,r({i})))")
        #             + E(f"( P({i},spenso::mink(4,l({i}))) * N(spenso::mink(4,r({i}))) + N(spenso::mink(4,l({i}))) * P({i},spenso::mink(4,r({i}))) )/ P({i},spenso::cind(0))")  # nopep8
        #             - E(f"( P({i},spenso::mink(4,l({i}))) * P({i},spenso::mink(4,r({i}))) ) / ( P({i},spenso::cind(0))^2 )")  # nopep8
        #         ),
        #         my_lib)
        #     pol_sum_i.execute(my_lib)
        #     pol_sum_i = pol_sum_i.result_tensor(my_lib)
        #     pol_sums.append(pol_sum_i)

        gamma = TensorStructure.gammadD(dim)

        g = TensorStructure.metric(mink)

        bis = Representation.bis(4)
        u, ubar, v, vbar, eps, epsbar = S(
            "alg::u", "alg::ubar", "alg::v", "alg::vbar", "alg::ϵ", "alg::ϵbar")
        i_, j_, d_, a_, b_ = S("i_", "j_", "d_", "a_", "b_")
        dummy = S("dummy")
        dummy_ss = S("dummy_ss")

        dot = S("symbolica_community::dot", is_linear=True, is_symmetric=True)

        expressions = {'left': left, 'right': right}
        lorentz_tensors = {}
        polarization_vecs = {'left': None, 'right': None}
        for i_side, (key, wrapper) in enumerate(zip(['left', 'right'], [l_side, r_side])):
            side_expression = E("0")
            for graph_id, graph_expr in sorted(expressions[key].items(), key=lambda x: x[0]):
                num, denom = self.build_graph_expression(
                    graph_expr, to_lmb=True, split_num_denom=True)
                if key == 'right':
                    num = conj(num)
                    denom = conj(denom)
                num = wrap_dummies(num, wrapper)
                num_pol_vectors_splits = self.expand_pol_vectors(num)
                if len(num_pol_vectors_splits) != 1:
                    raise MadLoop7Error(
                        "Only one polarization vector structure is expected to be found per process")
                pol_vectors = num_pol_vectors_splits[0][0]
                num = num_pol_vectors_splits[0][1]
                if polarization_vecs[key] is None:
                    polarization_vecs[key] = pol_vectors
                else:
                    if pol_vectors != polarization_vecs[key]:
                        raise MadLoop7Error(
                            "Polarization vector structures do not match between graphs")
                color_lorentz_splits = self.expand_color(num)
                for lor_struct_id, (col, lor) in enumerate(color_lorentz_splits):
                    lorentz_tensors[(i_side, graph_id, lor_struct_id)] = lor
                    side_expression = side_expression + \
                        S("color")(col) * \
                        S("denom")(denom) * \
                        S("T")(i_side, graph_id, lor_struct_id)

                # Not possible for now as these routines requires to simplify down to a scalar
                # num = simplify_color(num)
                # num = simplify_gamma(num)
            expressions[key] = side_expression

        assert (expressions['left'] is not None)
        assert (expressions['right'] is not None)

        if polarization_vecs['left'] is None:
            polarization_vecs['left'] = E("1")
        if polarization_vecs['right'] is None:
            polarization_vecs['right'] = E("1")
        assert (polarization_vecs['left'] is not None)
        assert (polarization_vecs['right'] is not None)

        pol_spin_sum_input = (
            polarization_vecs['left'] * polarization_vecs['right'])
        # For now only support massless externals
        if PHYSICAL_VECTOR_SUM_RULE:
            transverse_physical_vector_sum_num = n(i_, dummy_ss(i_, 1)) * p(i_, dummy_ss(i_, 1)) * (
                p(i_, l_side(a_)) * n(i_, r_side(a_)) +
                n(i_, l_side(a_)) * p(i_, r_side(a_))
            ) - \
                (
                n(i_, dummy_ss(i_, 2)) * n(i_, dummy_ss(i_, 2)) *
                p(i_, l_side(a_)) * p(i_, r_side(a_))
            )
            transverse_physical_vector_sum_denom_a = n(i_, dummy_ss(i_, 3)) * p(i_, dummy_ss(i_, 3)) * n(i_, dummy_ss(i_, 4)) * p(i_, dummy_ss(i_, 4))  # nopep8
            transverse_physical_vector_sum_denom_b = dot(N_symbol(i_), P_symbol(i_))**2  # nopep8
        else:
            transverse_physical_vector_sum_num = E("0")
            transverse_physical_vector_sum_denom_a = E("1")
            transverse_physical_vector_sum_denom_b = E("1")

        spin_sum_rules = []
        for sides in [(l_side, r_side), (r_side, l_side)]:
            spin_sum_rules.extend([
                (
                    eps(i_, mink(sides[0](a_)))*epsbar(i_, mink(sides[1](a_))),
                    (
                        -transverse_physical_vector_sum_denom_a * g(sides[0](a_), sides[1](a_)) + transverse_physical_vector_sum_num,  # nopep8
                        transverse_physical_vector_sum_denom_b
                    )
                ),
                (
                    vbar(i_, bis(sides[0](a_)))*v(i_, bis(sides[1](a_))),
                    (
                        gamma(sides[0](a_), sides[1](a_), dummy(i_, a_)) * p(i_, dummy(i_, a_)),  # nopep8
                        E("1")
                    )
                ),
                (
                    ubar(i_, bis(sides[0](a_)))*u(j_, bis(sides[1](a_))),
                    (
                        -gamma(sides[0](a_), sides[1](a_), dummy(i_, a_)) * p(i_, dummy(i_, a_)),  # nopep8
                        E("1")
                    )
                ),
            ])

        # print("expressions['left']=", expressions['left'])
        # print("expressions['right']=", expressions['right'])
        # print("polarization_vecs['left']=", polarization_vecs['left'])
        # print("polarization_vecs['right']=", polarization_vecs['right'])
        # print("pol_processed_bef=", pol_spin_sum_input)

        pol_spin_sum_num = pol_spin_sum_input
        pol_spin_sum_denom = pol_spin_sum_input
        for src, (trgt_num, trgt_denom) in spin_sum_rules:
            pol_spin_sum_num = pol_spin_sum_num.replace(
                src, trgt_num, repeat=True)
            pol_spin_sum_denom = pol_spin_sum_denom.replace(
                src, trgt_denom, repeat=True)

        # Define choice of gauge vectors
        # "temporal": n = (1,0,0,0)
        # "k-axial": n = (k^0, -k_vec)
        # GAUGE_VECTOR_CHOICE = "madgraph"
        GAUGE_VECTOR_CHOICE = "k-axial"

        for i in range(process.n_external):
            n_temporal_vector = sparse_empty(
                TensorStructure(mink4, E(f"{i}"), name=N_symbol), type(E("1")))
            match GAUGE_VECTOR_CHOICE:
                case "temporal":
                    n_temporal_vector[[0,]] = E("1")
                case "k-axial":
                    t = TensorNetwork(p4(i, 1)*TensorStructure.id(mink4)(1, 2))
                    t.execute(my_lib)
                    t = t.result_tensor(my_lib)
                    for t_idx in t.structure():
                        n_temporal_vector[t_idx] = t[t_idx]
                    # for j in range(4):
                    #     n_temporal_vector[[j,]] = t[[j,]]
            my_lib.register(n_temporal_vector)

        # Add dot products replacement
        dot_products_replacement = []
        for i in range(process.n_external):
            for j in range(process.n_external):
                if i == j:
                    dot_products_replacement.append((
                        dot(P_symbol(i), P_symbol(j)),
                        E("0")
                    ))
                elif i < j:
                    t = TensorNetwork.from_expression(
                        p4(i, 1)*p4(j, 1), my_lib)
                    t.execute(my_lib)
                    t = t.result_tensor(my_lib)
                    dot_products_replacement.append((dot(P_symbol(i), P_symbol(j)), t[[]]))  # nopep8

                t = TensorNetwork.from_expression(n4(i, 1)*n4(j, 1), my_lib)
                t.execute(my_lib)
                t = t.result_tensor(my_lib)
                dot_products_replacement.append((dot(N_symbol(i), N_symbol(j)), t[[]]))  # nopep8

                t = TensorNetwork.from_expression(n4(i, 1)*p4(j, 1), my_lib)
                t.execute(my_lib)
                t = t.result_tensor(my_lib)

                dot_products_replacement.append((dot(N_symbol(i), P_symbol(j)), t[[]]))  # nopep8

                t = TensorNetwork.from_expression(p4(i, 1)*n4(j, 1), my_lib)
                t.execute(my_lib)
                t = t.result_tensor(my_lib)
                dot_products_replacement.append((dot(P_symbol(i), N_symbol(j)), t[[]]))  # nopep8

        # Ready to process it whole
        def substitute_constants(input_e: Expression) -> Expression:
            constants = [
                (dim, E("4")),
                (E("alg::TR"), E("1/2")),
                (E("alg::Nc"), E("3")),
                (E("alg::CF"), E("4/3")),
                (E("alg::CA"), E("3")),
            ]
            for src, trgt in constants:
                input_e = input_e.replace(src, trgt)
            return input_e

        me = expressions['left'] * expressions['right']  # type: ignore

        # Unwrap color and simplify
        me = me.replace(E("color(x__)"), E("x__"))
        # me = me.replace(E("denom(x__)"), E("1"))
        # me = me.replace(E("T(x_,y_,z_)"), E("1"))
        # print("Before color simplification:", me)
        me = simplify_color(me)
        me = substitute_constants(me)

        # tmp = E("""((spenso::N(4,spenso::mink(4,python::l(2)))*spenso::P(4,spenso::mink(4,python::r(2)))+spenso::N(4,spenso::mink(4,python::r(2)))*spenso::P(4,spenso::mink(4,python::l(2))))*spenso::N(4,spenso::mink(4,python::dummy_ss(4,1)))*spenso::P(4,spenso::mink(4,python::dummy_ss(4,1)))+-1*spenso::N(4,spenso::mink(4,python::dummy_ss(4,2)))^2*spenso::P(4,spenso::mink(4,python::l(2)))*spenso::P(4,spenso::mink(4,python::r(2)))+-1*spenso::N(4,spenso::mink(4,python::dummy_ss(4,3)))*spenso::N(4,spenso::mink(4,python::dummy_ss(4,4)))*spenso::P(4,spenso::mink(4,python::dummy_ss(4,3)))*spenso::P(4,spenso::mink(4,python::dummy_ss(4,4)))*spenso::g(spenso::mink(4,python::l(2)),spenso::mink(4,python::r(2))))*((spenso::N(5,spenso::mink(4,python::l(3)))*spenso::P(5,spenso::mink(4,python::r(3)))+spenso::N(5,spenso::mink(4,python::r(3)))*spenso::P(5,spenso::mink(4,python::l(3))))*spenso::N(5,spenso::mink(4,python::dummy_ss(5,1)))*spenso::P(5,spenso::mink(4,python::dummy_ss(5,1)))+-1*spenso::N(5,spenso::mink(4,python::dummy_ss(5,2)))^2*spenso::P(5,spenso::mink(4,python::l(3)))*spenso::P(5,spenso::mink(4,python::r(3)))+-1*spenso::N(5,spenso::mink(4,python::dummy_ss(5,3)))*spenso::N(5,spenso::mink(4,python::dummy_ss(5,4)))*spenso::P(5,spenso::mink(4,python::dummy_ss(5,3)))*spenso::P(5,spenso::mink(4,python::dummy_ss(5,4)))*spenso::g(spenso::mink(4,python::l(3)),spenso::mink(4,python::r(3))))*(-1*spenso::G^2*spenso::P(0,spenso::mink(4,python::r(20)))*spenso::𝑖*spenso::𝟙(spenso::bis(4,python::r(0)),spenso::bis(4,python::r(7)))*spenso::𝟙(spenso::bis(4,python::r(1)),spenso::bis(4,python::r(4)))*spenso::𝟙(spenso::mink(4,python::r(2)),spenso::mink(4,python::r(5)))*spenso::𝟙(spenso::mink(4,python::r(3)),spenso::mink(4,python::r(4)))*weyl::gamma(spenso::mink(4,python::r(20)),spenso::bis(4,python::r(5)),spenso::bis(4,python::r(6)))*weyl::gamma(spenso::mink(4,python::r(4)),spenso::bis(4,python::r(4)),spenso::bis(4,python::r(5)))*weyl::gamma(spenso::mink(4,python::r(5)),spenso::bis(4,python::r(6)),spenso::bis(4,python::r(7)))+spenso::G^2*spenso::P(2,spenso::mink(4,python::r(20)))*spenso::𝑖*spenso::𝟙(spenso::bis(4,python::r(0)),spenso::bis(4,python::r(7)))*spenso::𝟙(spenso::bis(4,python::r(1)),spenso::bis(4,python::r(4)))*spenso::𝟙(spenso::mink(4,python::r(2)),spenso::mink(4,python::r(5)))*spenso::𝟙(spenso::mink(4,python::r(3)),spenso::mink(4,python::r(4)))*weyl::gamma(spenso::mink(4,python::r(20)),spenso::bis(4,python::r(5)),spenso::bis(4,python::r(6)))*weyl::gamma(spenso::mink(4,python::r(4)),spenso::bis(4,python::r(4)),spenso::bis(4,python::r(5)))*weyl::gamma(spenso::mink(4,python::r(5)),spenso::bis(4,python::r(6)),spenso::bis(4,python::r(7))))*(-1*spenso::P(2,spenso::mink(4,python::l(20)))+spenso::P(0,spenso::mink(4,python::l(20))))*-1*spenso::G^2*spenso::P(2,spenso::mink(4,python::dummy(2,0)))*spenso::P(3,spenso::mink(4,python::dummy(3,1)))*spenso::𝑖*spenso::𝟙(spenso::bis(4,python::l(0)),spenso::bis(4,python::l(7)))*spenso::𝟙(spenso::bis(4,python::l(1)),spenso::bis(4,python::l(4)))*spenso::𝟙(spenso::mink(4,python::l(2)),spenso::mink(4,python::l(5)))*spenso::𝟙(spenso::mink(4,python::l(3)),spenso::mink(4,python::l(4)))*weyl::gamma(spenso::bis(4,python::l(1)),spenso::bis(4,python::r(1)),spenso::mink(4,python::dummy(3,1)))*weyl::gamma(spenso::bis(4,python::r(0)),spenso::bis(4,python::l(0)),spenso::mink(4,python::dummy(2,0)))*weyl::gamma(spenso::mink(4,python::l(20)),spenso::bis(4,python::l(6)),spenso::bis(4,python::l(5)))*weyl::gamma(spenso::mink(4,python::l(4)),spenso::bis(4,python::l(5)),spenso::bis(4,python::l(4)))*weyl::gamma(spenso::mink(4,python::l(5)),spenso::bis(4,python::l(7)),spenso::bis(4,python::l(6)))""")
        # # tmp = "spenso::N(4,spenso::mink(4,python::l(2)))"
        # tmp = tmp.replace(E("weyl::gamma(spenso::mink(x__),y__)"), E(
        #     "weyl::gamma(y__,spenso::mink(x__))"))
        # tmp = cook_indices(tmp)
        # print(tmp.to_canonical_string())
        # my_lib = TensorLibrary.weyl()
        # # t = TensorNetwork.from_expression(tmp, my_lib)
        # t = TensorNetwork(tmp)
        # print("BEF", t)
        # t.execute(my_lib)
        # t = t.result_tensor(my_lib)
        # print("AFT", t)
        # stop
        # Now address Lorentz

        def process_lorentz(match):
            left_key = (0, int(str(match[S("LeftGraphID_")])), int(str(match[S("LeftTermID_")])))  # nopep8
            right_key = (1, int(str(match[S("LeftGraphID_")])), int(str(match[S("LeftTermID_")])))  # nopep8
            print(f"Doing T{left_key}*T{right_key}")
            left_tensor = lorentz_tensors[left_key]
            right_tensor = lorentz_tensors[right_key]
            tensor_expr = left_tensor * right_tensor * pol_spin_sum_num
            tensor_expr = tensor_expr.replace(
                E("spenso::mink(python::dim,x__)"), E("spenso::mink(4,x__)"))
            tensor_expr = tensor_expr.replace(
                E("alg::gamma(x___)"), E("weyl::gamma(x___)"))
            tensor_expr = tensor_expr.replace(E("weyl::gamma(spenso::mink(x__),y__)"), E(
                "weyl::gamma(y__,spenso::mink(x__))"))

            # HACK
            HACK_MODE_I = int(str(match[S("LeftGraphID_")])) + \
                int(str(match[S("LeftTermID_")]))
            tensor_expr = p4((HACK_MODE_I) % 3, 1)*p4((HACK_MODE_I+1) % 3, 1)

            tensor_expr = cook_indices(tensor_expr)
            # print("Processing:\n", tensor_expr.to_canonical_string())
            t = TensorNetwork.from_expression(tensor_expr, my_lib)
            t.execute(my_lib)
            # t = t.result_scalar(my_lib)
            t = t.result_tensor(my_lib)
            res = t[[]]
            # print("Resulting tensor:", res.to_canonical_string())
            # print("Resulting tensor:", res)
            # print("DONE!")

            return res

        me = me.replace(E("T(0,LeftGraphID_,LeftTermID_)*T(1,RightGraphID_,RightTermID_)"),
                        lambda m: process_lorentz(m))

        # print("Before Dot product substitution!\n", me)

        # Multiply in external polarization vectors spin-sum
        me = me * E("denom")(pol_spin_sum_denom)

        # Substitute dot products appearing in denominators now
        for src, trgt in dot_products_replacement:
            me = me.replace(src, trgt, repeat=True)

        # Unwrap denominators
        me = me.replace(E("denom(x_)"), E("1/x_"))

        # print("After Dot product substitution!\n", me)

        return me

    def build_matrix_element_expression_tensor_networks(self, process: HardCodedProcess, expressions: dict[Any, Any], mode: EvalMode) -> Any:

        self.import_symbolica_community()
        from symbolica_community import Expression, S, E  # type: ignore # nopep8

        match mode:
            case EvalMode.TREExTREE:
                me_expr = self.sum_square_left_right_tensor_networks(
                    process, expressions["tree"]["graph_expressions"], expressions["tree"]["graph_expressions"])
                me_expr = me_expr * E(process.overall_factor)
                return me_expr

            case EvalMode.LOOPxTREE:
                raise MadLoop7Error(
                    "Loop x tree matrix element expression building not implemented yet!")
            case EvalMode.LOOPxLOOP:
                raise MadLoop7Error(
                    "Loop x loop matrix element expression building not implemented yet!")

    def export_evaluator_tensor_networks(self, process: HardCodedProcess, matrix_element_expression: Any, mode: EvalMode) -> None:
        self.import_symbolica_community()
        from symbolica_community.tensors import TensorNetwork, Representation, TensorStructure, TensorLibrary  # type: ignore # nopep8
        from symbolica_community import Expression, S, E  # type: ignore # nopep8

        madloop7_output = os.path.abspath(pjoin(self.config["output_path"], MADLOOP7_OUTPUT_DIR, process.name))  # nopep8
        if not os.path.isdir(madloop7_output):
            os.makedirs(madloop7_output)

        mink4 = Representation.mink(4)
        P_symbol = S("spenso::P")
        P4 = TensorStructure(mink4, name=P_symbol)

        def p4(i, j):
            return P4(i, ';', j)

        my_lib = TensorLibrary.weyl()
        match mode:
            case EvalMode.TREExTREE:

                # Overall energy-momentum conservation
                # complement = E("0")
                # complement_n = E("0")
                # for i in range(process.n_external-1):
                #     if i < 2:
                #         complement = complement + P(i)
                #         complement_n = complement_n + N(i)
                #     else:
                #         complement = complement - P(i)
                #         complement_n = complement_n - N(i)
                # matrix_element_expression = matrix_element_expression.replace(
                #     P(process.n_external-1), complement)
                # matrix_element_expression = matrix_element_expression.replace(
                #     N(process.n_external-1), complement_n)
                kinematic_symbols = []
                for i in range(process.n_external):
                    t = TensorNetwork(p4(i, 1))
                    t.execute(my_lib)
                    t = t.result_tensor(my_lib)
                    for i_idx, lor_idx in enumerate(t):
                        kinematic_symbols.append(E(f"kin_p{i+1}_{i_idx}"))
                        matrix_element_expression = matrix_element_expression.replace(
                            lor_idx, kinematic_symbols[-1])

                # Do a cleaner filter eventually,
                model_param_symbols = [s for s in matrix_element_expression.get_all_symbols(
                    False) if s not in kinematic_symbols]
                model = get_model(process.model)
                model_parameters = model.get('parameter_dict')
                matrix_element_expression = matrix_element_expression.collect_factors()

                with open(pjoin(madloop7_output, "me_expression.txt"), "w") as f:
                    f.write(matrix_element_expression.to_canonical_string())
                shutil.copyfile(
                    pjoin(root_path, "templates", "phase_space_generator.py"),
                    pjoin(madloop7_output, "phase_space_generator.py"),
                )
                run_sa_template = open(
                    pjoin(root_path, "templates", "run_sa_tensor_networks.py"), 'r').read()
                replace_dict = {
                    'symbolica_community_path': self.config["symbolica_community_path"],
                    'n_externals': process.n_external,
                }
                model_param_values = []
                for p in model_param_symbols:
                    p_name = str(p).replace('spenso::', '')
                    if p_name not in model_parameters:
                        p_name = f'mdl_{p_name}'
                    model_param_values.append(
                        (p, model_parameters[p_name].value)
                    )
                replace_dict['model_parameters'] = ',\n'.join(
                    [f"(S(\"{p.to_canonical_string()}\"), {v})" for (p, v) in model_param_values])

                with open(pjoin(madloop7_output, "run.py"), "w") as f:
                    f.write(run_sa_template % replace_dict)

            case EvalMode.LOOPxTREE:
                raise MadLoop7Error(
                    "Loop x tree matrix element expression evaluation not implemented yet!")
            case EvalMode.LOOPxLOOP:
                raise MadLoop7Error(
                    "Loop x loop matrix element expression evaluation not implemented yet!")

    def import_symbolica_community(self) -> None:

        if self.config["symbolica_community_path"] is not None:
            sc_path = os.path.abspath(
                pjoin(self.config["symbolica_community_path"], "python"))
            if sc_path not in sys.path:
                sys.path.insert(0, sc_path)
            try:
                import symbolica_community as sc  # type: ignore # nopep8
            except:
                raise MadLoop7Error('\n'.join([
                    "ERROR: Could not import Python's symbolica_community module from {}".format(
                        self.config["symbolica_community_path"]),
                    "Add '<SYMBOLICA_COMMUNITY_INSTALLATION_DIRECTORY>' to your PYTHONPATH or specify it under 'symbolica_community_path' in the configuration file used to load MadLoop7.",]))
        else:
            try:
                import symbolica_community as sc  # type: ignore # nopep8
            except:
                raise MadLoop7Error('\n'.join([
                    "ERROR: Could not import Python's symbolica_community module.",
                    "Add '<SYMBOLICA_COMMUNITY_INSTALLATION_DIRECTORY>/python' to your PYTHONPATH or specify it under 'symbolica_community_path' in the configuration file used to load MadLoop7.",]))

    def import_gammaloop(self) -> None:
        if self.config["gammaloop_path"] is not None:
            sc_path = os.path.abspath(
                pjoin(self.config["gammaloop_path"], "python"))
            if sc_path not in sys.path:
                sys.path.insert(0, sc_path)
            try:
                from gammaloop.interface.gammaloop_interface import GammaLoop, CommandList  # type: ignore # nopep8
            except:
                raise MadLoop7Error('\n'.join([
                    "ERROR: Could not import Python's gammaloop module from {}".format(
                        self.config["symbolica_community_path"]),
                    "Add '<SYMBOLICA_COMMUNITY_INSTALLATION_DIRECTORY>' to your PYTHONPATH or specify it under 'gammaloop_path' in the configuration file used to load MadLoop7.",]))
        else:
            try:
                from gammaloop.interface.gammaloop_interface import GammaLoop, CommandList  # type: ignore # nopep8
            except:
                raise MadLoop7Error('\n'.join([
                    "ERROR: Could not import Python's gammaloop module.",
                    "Add '<SYMBOLICA_COMMUNITY_INSTALLATION_DIRECTORY>/python' to your PYTHONPATH or specify it under 'gammaloop_path' in the configuration file used to load MadLoop7.",]))

    def generate(self, process_name: str = "NOT_SPECIFIED", tree_graph_ids: list[int] | None = None, loop_graph_ids: list[int] | None = None, evaluation_strategy: str | None = None, **_opts) -> Any:

        if evaluation_strategy is None:
            madloop7_evaluation_strategy = EvaluationStrategy.TENSOR_NETWORKS
        else:
            madloop7_evaluation_strategy = EvaluationStrategy.from_string(
                evaluation_strategy)

        for dir in [MADGRAPH_OUTPUT_DIR, graphs_output_DIR, GAMMALOOP_OUTPUT_DIR, MADLOOP7_OUTPUT_DIR]:
            if not os.path.isdir(pjoin(self.config["output_path"], dir)):
                os.makedirs(pjoin(self.config["output_path"], dir))

        if process_name not in HARDCODED_PROCESSES:
            raise MadLoop7Error(f"Process {process_name} not supported")
        process = HARDCODED_PROCESSES[process_name]

        # Generate graphs if the madsymbolic output is not already present
        if any(not os.path.isfile(pjoin(self.config["output_path"], GAMMALOOP_OUTPUT_DIR, f"{process.name}_{graph_class}.dot")) for graph_class in process.get_graph_categories()):
            self.generate_process(process)

        # Now process the graph with gammaLoop and spenso to get the symbolic expression
        if any(not os.path.isdir(pjoin(self.config["output_path"], GAMMALOOP_OUTPUT_DIR, f"{process.name}_{graph_class}")) for graph_class in process.get_graph_categories()):
            self.build_expressions_with_gammaloop(process)

        if not os.path.isdir(pjoin(self.config["output_path"], MADLOOP7_OUTPUT_DIR, f"{process.name}")):
            self.build_evaluators_with_spenso(
                process, tree_graph_ids, loop_graph_ids, madloop7_evaluation_strategy)

        logger.info(f"Process generation for {process_name} completed.")

    def clean(self, process_name: str) -> None:
        for graph_class in ['', '_tree', '_loop']:
            output_name = f"{process_name}{graph_class}"
            for dir in [MADGRAPH_OUTPUT_DIR, graphs_output_DIR, GAMMALOOP_OUTPUT_DIR, MADLOOP7_OUTPUT_DIR]:
                process_dir = pjoin(
                    self.config["output_path"], dir, output_name)
                if os.path.isdir(process_dir):
                    shutil.rmtree(process_dir)
            if os.path.isfile(pjoin(self.config["output_path"], GAMMALOOP_OUTPUT_DIR, f"{output_name}.dot")):
                os.remove(
                    pjoin(self.config["output_path"], GAMMALOOP_OUTPUT_DIR, f"{output_name}.dot"))
            if os.path.isfile(pjoin(self.config["output_path"], MADGRAPH_OUTPUT_DIR, f"{output_name}.mg5")):
                os.remove(
                    pjoin(self.config["output_path"], MADGRAPH_OUTPUT_DIR, f"{output_name}.mg5"))
        logger.info(f"Cleaned outputs for process {process_name}.")

    def install_dependencies(self, symbolica_community: bool = False) -> None:
        logger.critical("Dependency installation not implemented yet")

    def playground(self) -> None:
        import random

        # Example on how to get a quick python model to get all the numerical values of the model parameters
        sm_model = get_model("sm")
        # print(pformat(sm_model.parameters_dict))
        print("mdl_Gf=", sm_model.parameters.mdl_Gf)

        # And how to generate a random 2 > 2 PS point
        E_cm = 1000.0
        ps_generator = FlatPhaseSpaceGenerator(
            [0.]*2, [0.]*3,
            beam_Es=(E_cm/2., E_cm/2.),
            beam_types=(0, 0)
        )
        xs = [random.random() for _ in range(ps_generator.nDimPhaseSpace())]
        ps_point, jacobian, _x1, _x2 = ps_generator.get_PS_point(xs)

        print(f"Random phase-space point:\n{str(ps_point)}\nwith jacobian: {jacobian:f}")  # nopep8
