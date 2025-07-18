import multiprocessing.pool
from . import logger
from pprint import pformat, pprint
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
import re
import multiprocessing
from enum import Enum

from pprint import pformat, pprint

root_path = os.path.abspath(os.path.dirname(__file__))

pjoin = os.path.join

graphs_output_DIR = "graphs_outputs"
MADGRAPH_OUTPUT_DIR = "madgraph_outputs"
GAMMALOOP_OUTPUT_DIR = "gammaloop_outputs"
MADLOOP7_OUTPUT_DIR = "madloop7_outputs"

GAUGE_VECTOR_SQUARED_IS_ZERO = True
END_CHAR = '\r'
# END_CHAR = '\n'


def import_symbolica_community() -> None:
    symbolica_community_path = os.environ.get(
        "SYMBOLICA_COMMUNITY_PATH", None)
    if symbolica_community_path is not None:
        sc_path = os.path.abspath(
            pjoin(symbolica_community_path, "python"))
        if sc_path not in sys.path:
            sys.path.insert(0, sc_path)
        try:
            import symbolica_community as sc  # type: ignore # nopep8
        except:
            raise MadLoop7Error('\n'.join([
                "ERROR: Could not import Python's symbolica_community module from {}".format(
                    symbolica_community_path),
                "Add '<SYMBOLICA_COMMUNITY_INSTALLATION_DIRECTORY>' to your PYTHONPATH or specify it under 'symbolica_community_path' in the configuration file used to load MadLoop7.",]))
    else:
        try:
            import symbolica_community as sc  # type: ignore # nopep8
        except:
            raise MadLoop7Error('\n'.join([
                "ERROR: Could not import Python's symbolica_community module.",
                "Add '<SYMBOLICA_COMMUNITY_INSTALLATION_DIRECTORY>/python' to your PYTHONPATH or specify it under 'symbolica_community_path' in the configuration file used to load MadLoop7.",]))


import_symbolica_community()  # nopep8
import symbolica_community as sc  # type: ignore # nopep8


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

    def __init__(self, config_path: str, **opts) -> None:
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.config.update(opts)

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

        def evaluate_graph_overall_factor(overall_factor: Any) -> Any:
            for header in ["AutG",
                           "CouplingsMultiplicity",
                           "InternalFermionLoopSign",
                           "ExternalFermionOrderingSign",
                           "AntiFermionSpinSumSign",
                           "NumeratorIndependentSymmetryGrouping"]:
                overall_factor = overall_factor.replace(
                    sc.E(f"{header}(x_)"), sc.E("x_"), repeat=True)
            overall_factor = overall_factor.replace(
                sc.E("NumeratorDependentGrouping(GraphId_,ratio_,GraphSymmetryFactor_)"), sc.E("ratio_*GraphSymmetryFactor_"), repeat=True)
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
            for graph_expr_path in os.listdir(pjoin(
                    self.config["output_path"], GAMMALOOP_OUTPUT_DIR, output_name, "sources",
                    output_metadata["output_type"], output_metadata["contents"][0], "expressions")):
                # print(graph_expr_path)
                graph_id = int(graph_expr_path.split("_")[-2])
                graph_ids = tree_graph_ids if graph_class == "tree" else loop_graph_ids
                if graph_ids is None or graph_id in graph_ids:
                    with open(pjoin(self.config["output_path"], GAMMALOOP_OUTPUT_DIR, output_name, "sources", output_metadata["output_type"], output_metadata["contents"][0], "expressions", graph_expr_path), "r") as f:
                        expr = json.load(f)
                        graph_expressions[graph_id] = {
                            'expression': expr[0],
                            'overall_factor': evaluate_graph_overall_factor(sc.E(expr[1])),
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

        match evaluation_strategy:
            case EvaluationStrategy.ONLY_DOT_PRODUCTS:
                logger.info("Building evaluator with only dot products...")
                matrix_element_terms = self.build_matrix_element_expression_only_dots(
                    process, expressions,  eval_mode)
                self.export_evaluator_only_dots(
                    process, matrix_element_terms, eval_mode)
            case EvaluationStrategy.TENSOR_NETWORKS:
                logger.info("Building evaluator with tensor networks...")
                matrix_element_expression, overall_emr_replacements = self.build_matrix_element_expression_tensor_networks(
                    process, expressions,  eval_mode)
                self.export_evaluator_tensor_networks(
                    process, matrix_element_expression, overall_emr_replacements, eval_mode)

    def build_graph_expression(self, graph_id, graph_expr: dict[Any, Any], to_lmb: bool = False) -> Any:
        """Build the graph expression from the graph expression dictionary."""

        def E_sp(expr: str) -> sc.Expression:
            """Create a symbolica_community expression with the spenso namespace."""
            return sc.E(expr, default_namespace="spenso")

        def curate(expr: sc.Expression) -> sc.Expression:
            expr = expr.replace(E_sp("Metric(x_,y_)"),
                                E_sp("g(x_,y_)"), repeat=True)
            expr = expr.replace(E_sp("id(x_,y_)"),
                                E_sp("g(x_,y_)"), repeat=True)
            expr = expr.replace(sc.E("spenso::γ(x_,y_,z_)"), sc.E(
                "spenso::gamma(x_,y_,z_)"), repeat=True)
            expr = expr.replace(sc.E("spenso::T(x_,y_,z_)"),
                                sc.E("spenso::t(x_,y_,z_)"), repeat=True)
            expr = expr.replace(sc.E("spenso::mink(4,x_)"), sc.E(
                "spenso::mink(python::dim,x_)"), repeat=True)
            expr = expr.replace(sc.E("spenso::coad(8,x_)"), sc.E(
                "spenso::coad(spenso::Nc^2-1,x_)"), repeat=True)
            expr = expr.replace(sc.E("spenso::cof(3,x_)"), sc.E(
                "spenso::cof(spenso::Nc,x_)"), repeat=True)
            expr = expr.replace(sc.E("spenso::gamma(spenso::mink(x__),y__)"), sc.E(
                "spenso::gamma(y__,spenso::mink(x__))"))
            expr = expr.replace(sc.E("spenso::T(spenso::coad(x__),spenso::cof(y__),spenso::dind(cof(z__)))"), sc.E(
                "spenso::T(spenso::coad(x__),spenso::cof(y__),spenso::dind(cof(z__)))"))
            expr = expr.replace(sc.E("spenso::𝑖"), sc.E("1𝑖"))
            return expr

        numerator = curate(
            E_sp(graph_expr['expression']))*graph_expr['overall_factor']
        denominator = sc.E("1")
        for (p, m) in graph_expr['denominators'].items():
            denominator = denominator * \
                (sc.S("spenso::dot", is_linear=True,
                 is_symmetric=True)(E_sp(p), E_sp(p)) - E_sp(m)**2)

        emr_to_lmb_replacements = {}
        if to_lmb:
            for (src, trgt) in graph_expr['momenta'].items():
                numerator = numerator.replace(
                    E_sp(src), E_sp(trgt), repeat=True)
                denominator = denominator.replace(
                    E_sp(src), E_sp(trgt), repeat=True)
        else:
            numerator = numerator.replace(
                E_sp("Q(i_,x___)"), E_sp("Q(EMRID(%d,i_),x___)" % graph_id))
            denominator = denominator.replace(
                E_sp("Q(i_,x___)"), E_sp("Q(EMRID(%d,i_),x___)" % graph_id))

            for (src, trgt) in graph_expr['momenta'].items():
                emr_to_lmb_replacements[E_sp(src).replace(E_sp("Q(i_,x___)"), E_sp("Q(EMRID(%d,i_),x___)" % graph_id))] = E_sp(trgt)  # nopep8

        return numerator, denominator, emr_to_lmb_replacements

    def sum_square_left_right_only_dots(self, left: dict[Any, Any], right: dict[Any, Any]) -> Any:
        """Combine left and right expressions into a single expression."""

        expressions = {'left': left, 'right': right}

        dim = sc.S("python::dim")
        mink = sc.tensors.Representation.mink(dim)
        bis = sc.tensors.Representation.bis(dim)
        P_symbol = sc.S("spenso::P")
        N_symbol = sc.S("spenso::N")
        Q = sc.tensors.TensorStructure(mink, name=sc.S("spenso::Q"))
        P = sc.tensors.TensorStructure(mink, name=P_symbol)
        N = sc.tensors.TensorStructure(mink, name=N_symbol)

        def q(i, j):
            return Q(i, ';', j)

        def p(i, j):
            return P(i, ';', j)

        def n(i, j):
            return N(i, ';', j)

        hep_lib = sc.tensors.TensorLibrary.hep_lib()

        def gamma(i, j, k):
            tt = f"spenso::gamma(spenso::bis(4,{i.to_canonical_string()}),spenso::bis(4,{j.to_canonical_string()}),spenso::mink(4,{k.to_canonical_string()}))"  # nopep8
            return sc.E(tt)

        def g(i, j):
            tt = f"spenso::g(spenso::mink(4,{i.to_canonical_string()}),spenso::mink(4,{j.to_canonical_string()}))"  # nopep8
            return sc.E(tt)

        bis = sc.tensors.Representation.bis(4)
        u, ubar, v, vbar, eps, epsbar = sc.S(
            "spenso::u", "spenso::ubar", "spenso::v", "spenso::vbar", "spenso::ϵ", "spenso::ϵbar")
        i_, j_, d_, a_, b_ = sc.S("i_", "j_", "d_", "a_", "b_")
        dummy = sc.S("dummy")
        l_side = sc.S("l")
        r_side = sc.S("r")
        dot = sc.S("spenso::dot", is_linear=True, is_symmetric=True)
        denom_marker = sc.S("denom")

        SIMPLIFY_INDIVIDUAL_INTERFERENCE_TERMS_SEPARATELY = True

        overall_emr_replacements = {}
        terms = []

        logger.info("Loading graph expressions%s..." % (
            (" and gamma simplifying amplitudes" if self.config['simplify_amplitude_gamma_algebra'] else "")
        ))
        if not SIMPLIFY_INDIVIDUAL_INTERFERENCE_TERMS_SEPARATELY:
            pol_vectors = {'left': None, 'right': None}
            for key in ['left', 'right']:
                side_expression = sc.E("0")
                for graph_id, graph_expr in sorted(expressions[key].items(), key=lambda x: x[0]):
                    num, denom, emr_to_lmb_replacements = self.build_graph_expression(graph_id,
                                                                                      graph_expr, to_lmb=(not self.config['work_in_emr']))
                    num = sc.algebraic_simplification.wrap_dummies(num.collect_num(), l_side if key == 'left' else r_side)  # nopep8
                    num_exp = self.expand_pol_vectors(num)
                    if len(num_exp) != 1:
                        raise MadLoop7Error(
                            "Expected only one term in the pol. vec. expansion of the squared matrix element, got %d terms." % (len(num_exp)))  # nopep8
                    pol_exp = num_exp[0][0]  # nopep8
                    if key == 'right':
                        pol_exp = sc.algebraic_simplification.conj(pol_exp)  # nopep8
                    if pol_vectors[key] is None:
                        pol_vectors[key] = pol_exp
                    else:
                        if pol_vectors[key] != pol_exp:
                            raise MadLoop7Error("Expected the same polarization vector for all squared matrix elements, got %s and %s." % (pol_vectors[key].to_canonical_string(), pol_exp.to_canonical_string()))  # type: ignore # nopep8
                    num = sc.algebraic_simplification.simplify_metrics(
                        num_exp[0][1])
                    if self.config['simplify_amplitude_gamma_algebra']:
                        num = sc.algebraic_simplification.simplify_gamma(num)
                    num = num.collect_num()

                    overall_emr_replacements.update(emr_to_lmb_replacements)
                    side_expression = side_expression + num / denom
                expressions[key] = side_expression

            if pol_vectors['left'] is None:
                pol_vectors['left'] = sc.E("1")
            if pol_vectors['right'] is None:
                pol_vectors['right'] = sc.E("1")
            left_e = expressions['left']
            right_e = sc.algebraic_simplification.conj(expressions['right'])

            terms.append(
                (
                    (-1, -1),
                    (left_e*right_e*pol_vectors['left']*pol_vectors['right']),
                    sc.E("1")
                )
            )
        else:
            pol_vectors = {'left': None, 'right': None}
            diagram_expressions = {'left': [], 'right': []}  # type: ignore # nopep8
            for graph_id, graph_expr in sorted(expressions['left'].items(), key=lambda x: x[0]):
                num, denom, emr_to_lmb_replacements = self.build_graph_expression(
                    graph_id, graph_expr, to_lmb=(not self.config['work_in_emr']))
                num = sc.algebraic_simplification.wrap_dummies(num.collect_num(), l_side)  # nopep8

                num_exp = self.expand_pol_vectors(num)
                if len(num_exp) != 1:
                    raise MadLoop7Error(
                        "Expected only one term in the pol. vec. expansion of the squared matrix element, got %d terms." % (len(num_exp)))  # nopep8
                pol_exp = num_exp[0][0]  # nopep8
                if pol_vectors['left'] is None:
                    pol_vectors['left'] = pol_exp
                else:
                    if pol_vectors['left'] != pol_exp:
                        raise MadLoop7Error("Expected the same polarization vector for all squared matrix elements, got %s and %s." % (pol_vectors['left'].to_canonical_string(), pol_exp.to_canonical_string()))  # type: ignore # nopep8
                num = sc.algebraic_simplification.simplify_metrics(
                    num_exp[0][1])

                if self.config['simplify_amplitude_gamma_algebra']:
                    # print("BEFORE GAMMA SIMPLIFICATION\n%s" % num.to_canonical_string())  # nopep8
                    num = sc.algebraic_simplification.simplify_gamma(num)

                num = num.collect_num()
                # print("AFTER GAMMA SIMPLIFICATION\n%s" % num.to_canonical_string())  # nopep8

                overall_emr_replacements.update(emr_to_lmb_replacements)
                diagram_expressions['left'].append(
                    (
                        graph_id,
                        num,
                        denom
                    )
                )
            # print(sorted(expressions['left'].items(), key=lambda x: x[0]))

            for graph_id, graph_expr in sorted(expressions['right'].items(), key=lambda x: x[0]):
                num, denom, emr_to_lmb_replacements = self.build_graph_expression(
                    graph_id, graph_expr, to_lmb=(not self.config['work_in_emr']))
                num = sc.algebraic_simplification.wrap_dummies(num.collect_num(), r_side)  # nopep8

                num_exp = self.expand_pol_vectors(num)
                if len(num_exp) != 1:
                    raise MadLoop7Error(
                        "Expected only one term in the pol. vec. expansion of the squared matrix element, got %d terms." % (len(num_exp)))  # nopep8
                pol_exp = sc.algebraic_simplification.conj(num_exp[0][0])  # nopep8
                if pol_vectors['right'] is None:
                    pol_vectors['right'] = pol_exp
                else:
                    if pol_vectors['right'] != pol_exp:
                        raise MadLoop7Error("Expected the same polarization vector for all squared matrix elements, got %s and %s." % (pol_vectors['right'].to_canonical_string(), pol_exp.to_canonical_string()))  # type: ignore # nopep8
                num = sc.algebraic_simplification.simplify_metrics(
                    num_exp[0][1])

                if self.config['simplify_amplitude_gamma_algebra']:
                    # print("BEFORE GAMMA SIMPLIFICATION\n%s" % num.to_canonical_string())  # nopep8
                    num = sc.algebraic_simplification.simplify_gamma(num)

                num = num.collect_num()
                # print("AFTER GAMMA SIMPLIFICATION\n%s" % num.to_canonical_string())  # nopep8

                overall_emr_replacements.update(emr_to_lmb_replacements)
                diagram_expressions['right'].append(
                    (
                        graph_id,
                        sc.algebraic_simplification.conj(num),
                        denom
                    )
                )

            # print(sorted(expressions['right'].items(), key=lambda x: x[0])[0])

            if pol_vectors['left'] is None:
                pol_vectors['left'] = sc.E("1")
            if pol_vectors['right'] is None:
                pol_vectors['right'] = sc.E("1")

            # print(diagram_expressions['right'])
            for (left_id, left_expr_num, left_expr_denom) in diagram_expressions['left']:
                for (right_id, right_expr_num, right_expr_denom) in diagram_expressions['right']:
                    # print("LEFT ID: %d, RIGHT ID: %d" % (left_id, right_id))  # nopep8
                    # print("LEFT")
                    # print(left_expr_num)
                    # print("RIGHT")
                    # print(right_expr_num)
                    # VHHACK
                    # if left_id == right_id:
                    #     continue
                    prod = left_expr_num*right_expr_num
                    terms.append(
                        (
                            (left_id, right_id),
                            prod*pol_vectors['left']*pol_vectors['right'],
                            left_expr_denom * right_expr_denom
                        )
                    )

        # Merge identical emr vectors across graphs
        unique_emrs = {}
        emrs_to_merge = {}
        for emr_v, lmb_expr in overall_emr_replacements.items():
            if lmb_expr not in emrs_to_merge:
                unique_emrs[emr_v] = lmb_expr
                emrs_to_merge[lmb_expr] = [emr_v,]
            else:
                emrs_to_merge[lmb_expr].append(emr_v)

        emr_to_lmb_replacements = unique_emrs

        emr_merging_replacements = []
        for emr in emrs_to_merge.values():
            emr_merging_replacements.extend([sc.Replacement(emr_v, emr[0]) for emr_v in emr[1:]])  # type: ignore # nopep8
        if len(emr_merging_replacements) > 0:
            logger.info("Merging %d EMR vectors" %
                        len(emr_merging_replacements))
            terms = [
                (
                    a_t[0],
                    a_t[1].replace_multiple(emr_merging_replacements),
                    a_t[2].replace_multiple(emr_merging_replacements),
                ) for a_t in terms
            ]

        if not GAUGE_VECTOR_SQUARED_IS_ZERO:
            raise NotImplementedError(
                "Dot product evaluation strategy only implemented for gauge vector norms equal to zero.")
        # For now only support massless externals
        if self.config['physical_vector_polarization_sum']:
            transverse_physical_vector_sum = (
                p(i_, l_side(a_)) * n(i_, r_side(a_)) +
                n(i_, l_side(a_)) * p(i_, r_side(a_))
            ) * denom_marker(dot(N_symbol(i_), P_symbol(i_)))
            if not GAUGE_VECTOR_SQUARED_IS_ZERO:
                transverse_physical_vector_sum = transverse_physical_vector_sum - \
                    (
                        dot(N_symbol(i_), N_symbol(i_)) *
                        p(i_, l_side(a_)) * p(i_, r_side(a_))
                    ) * denom_marker(dot(N_symbol(i_), P_symbol(i_)) ** 2)
        else:
            transverse_physical_vector_sum = sc.E("0")

        spin_sum_rules = []
        for sides in [(l_side, r_side), (r_side, l_side)]:
            spin_sum_rules.extend([
                (
                    eps(i_, mink(sides[0](a_)))*epsbar(i_, mink(sides[1](a_))),
                    -g(sides[0](a_), sides[1](a_)) + transverse_physical_vector_sum  # nopep8
                ),
                (
                    vbar(i_, bis(sides[0](a_)))*v(i_, bis(sides[1](a_))),
                    gamma(sides[0](a_), sides[1](a_), dummy(i_, a_)) * p(i_, dummy(i_, a_)),  # nopep8
                ),
                (
                    ubar(i_, bis(sides[0](a_)))*u(j_, bis(sides[1](a_))),
                    -gamma(sides[0](a_), sides[1](a_), dummy(i_, a_)) * p(i_, dummy(i_, a_))  # nopep8
                ),
            ])

        emr_replacements = [sc.Replacement(k, v)
                            for k, v in overall_emr_replacements.items()]
        final_result = []
        term_sizes = []
        logger.info(
            "Starting the processing of %d interference terms..." % len(terms))

        if self.config['n_cores'] <= 1:
            for i_t, (term_id, t, t_denom) in enumerate(terms):
                final_result.append(
                        MadLoop7.process_term_only_dots((term_id, i_t, t, t_denom, spin_sum_rules, emr_replacements, len(terms), True, False))[1])  # nopep8
        else:
            serialized_spin_sum_rules = [(ssr[0].to_canonical_string(), ssr[1].to_canonical_string()) for ssr in spin_sum_rules]  # nopep8
            serialized_emr_replacements = [(k.to_canonical_string(), v.to_canonical_string()) for k, v in overall_emr_replacements.items()]  # nopep8
            with multiprocessing.Pool(self.config['n_cores']) as pool:
                n_completed = 0
                for res in pool.imap_unordered(
                    MadLoop7.process_term_only_dots,
                    (
                        (term_id, i_t, t.to_canonical_string(), t_denom.to_canonical_string(
                        ), serialized_spin_sum_rules, serialized_emr_replacements, len(terms), False, True)
                        for i_t, (term_id,  t, t_denom) in enumerate(terms)  # nopep8
                    ),
                    chunksize=1
                ):
                    last_term_id = res[0]
                    res = sc.E(res[1])
                    n_completed += 1
                    final_result.append(res)
                    this_res_size = res.get_byte_size()
                    term_sizes.append(this_res_size)
                    term_sizes = sorted(term_sizes)
                    min_s, med_s, max_s = term_sizes[0], term_sizes[len(term_sizes)//2], term_sizes[-1]  # nopep8
                    total = sum(term_sizes)
                    print("Processed %-6d / %d terms. Term sizes: min=%-10dB, med=%-10dB, max=%-10dB | last_term=%-10s with size %-10dB | Total=%-10dB\r" % (
                        n_completed, len(terms), min_s, med_s, max_s, last_term_id, this_res_size, total), end='\r')  # nopep8

        return final_result

    @ staticmethod
    def substitute_constants(input_e: Any) -> Any:

        dim = sc.S("python::dim")
        constants = [
            (dim, sc.E("4")),
            (sc.E("spenso::TR"), sc.E("1/2")),
            (sc.E("spenso::Nc"), sc.E("3")),
            (sc.E("spenso::CF"), sc.E("4/3")),
            (sc.E("spenso::CA"), sc.E("3")),
        ]
        for src, trgt in constants:
            input_e = input_e.replace(src, trgt)
        return input_e

    @ staticmethod
    def process_term_only_dots(args) -> Any:
        """Process a single term."""
        term_id, i_t, t, t_denom, external_spin_sum_rules, emr_replacements_rules, n_terms, monitor_progress, serialized_io = args  # nopep8

        if serialized_io:
            t = sc.E(t)
            t_denom = sc.E(t_denom)
            external_spin_sum_rules = [(sc.E(ssr[0]), sc.E(ssr[1]))
                                       for ssr in external_spin_sum_rules]
            emr_replacements_rules = [sc.Replacement(
                sc.E(k), sc.E(v)) for k, v in emr_replacements_rules]

        denom_marker = sc.S("denom")
        dim = sc.S("python::dim")

        # progress_line = "Processing terms %-6d / %d : %-100s"
        progress_line = "Processing terms %-6d / %d : %-60s | expr size: %-10d bytes"

        if monitor_progress:
            print(progress_line % (i_t, n_terms, "start", t.get_byte_size()), end=END_CHAR)  # nopep8

        for src, trgt in external_spin_sum_rules:
            t = t.replace(src, trgt, repeat=True)

        t = t.replace(dim, sc.E("4"), repeat=True)

        t_input = t
        # print('NOW DOING COLOR FOR:\n%s' % (t.to_canonical_string()))  # nopep8

        col_expanded_t = MadLoop7.expand_color(t)
        t = sc.E("0")
        for i_step, (col, lor) in enumerate(col_expanded_t):
            if monitor_progress:
                print(progress_line % (i_t, n_terms,
                "Simplifying color, piece #%d / %d ..." % (i_step+1, len(col_expanded_t)), t.get_byte_size()), end=END_CHAR)  # nopep8
            t += lor*sc.algebraic_simplification.simplify_color(col)

        # This is useful for efficiency purposes
        t = MadLoop7.substitute_constants(t)
        t = t.collect_num()
        # print('AA')
        # print(t.to_canonical_string())
        # print('BB')
        # print(sc.algebraic_simplification.simplify_gamma(t).to_canonical_string())
        # print('CC')
        # print(sc.algebraic_simplification.simplify_gamma(sc.algebraic_simplification.simplify_gamma(t)).to_canonical_string())
        # stop

        # Verify that there no indices left
        for idx in [sc.E("spenso::coad(x___)"), sc.E("spenso::cof(x___)")]:
            try:
                _ = next(t.match(idx))
            except StopIteration:
                continue
            raise MadLoop7Error("Found remaining index of type %s in the final expression:\n%s\n Started with:\n%s" % (
                idx.to_canonical_string(),
                t.to_canonical_string(),
                t_input.to_canonical_string()
            ))

        # print('AFT:\n%s' % (t.to_canonical_string()))  # nopep8
        if monitor_progress:
            print(progress_line % (i_t, n_terms, "Simplifying metrics...", t.get_byte_size()), end=END_CHAR)  # nopep8
        t = sc.algebraic_simplification.simplify_metrics(t)

        if monitor_progress:
            print(progress_line % (i_t, n_terms, "Simplifying gamma algebra...", t.get_byte_size()), end=END_CHAR)  # nopep8
        t = sc.algebraic_simplification.simplify_gamma(t)
        # i_simplify = 1
        # while True:
        #     t_next = sc.algebraic_simplification.simplify_gamma(t)
        #     if t_next == t:
        #         break
        #     i_simplify += 1
        #     logger.warning("Gamma algebra simplification did not immediately converge, iterative step #%d" % i_simplify)  # nopep8
        #     t = t_next

        if monitor_progress:
            print(progress_line % (i_t, n_terms, "Converting to dot products...", t.get_byte_size()), end=END_CHAR)  # nopep8
        t = sc.algebraic_simplification.to_dots(t)

        # Verify that there no indices left
        for idx in [sc.E("spenso::mink(x___)"), sc.E("spenso::bis(x___)")]:
            try:
                _ = next(t.match(idx))
            except StopIteration:
                continue
            raise MadLoop7Error("Found remaining index of type %s in the final expression:\n%s\n Started with:\n%s" % (
                idx.to_canonical_string(),
                t.to_canonical_string(),
                t_input.to_canonical_string()
            ))

        if monitor_progress:
            print(progress_line % (i_t, n_terms, "EMR to LMB replacement...", t.get_byte_size()), end=END_CHAR)  # nopep8
        t = t.replace_multiple(emr_replacements_rules)

        if monitor_progress:
            print(progress_line % (i_t, n_terms, "Expansion of LMB result to combine terms...", t.get_byte_size()), end=END_CHAR)  # nopep8
        t = t.expand()

        if monitor_progress:
            print(progress_line % (i_t, n_terms, "Collect common coefficients...", t.get_byte_size()), end=END_CHAR)  # nopep8
        t = t.collect_num()

        if monitor_progress:
            print(progress_line % (i_t, n_terms, "Multiplying in denominators...", t.get_byte_size()), end=END_CHAR)  # nopep8
        t = t.replace(denom_marker(sc.E("x___")), 1/sc.E("x___"))
        t_denom = sc.algebraic_simplification.to_dots(t_denom)
        t_denom = t_denom.replace_multiple(emr_replacements_rules)

        t = t / t_denom

        if monitor_progress:
            print(progress_line % (i_t, n_terms, "Substituting in constants...", t.get_byte_size()), end=END_CHAR)  # nopep8

        t = MadLoop7.substitute_constants(t)

        # Verify that there no indices left
        for idx in [sc.E("spenso::mink(x___)"), sc.E("spenso::bis(x___)"), sc.E("spenso::coad(x___)"), sc.E("spenso::cof(x___)")]:
            try:
                _ = next(t.match(idx))
            except StopIteration:
                continue
            raise MadLoop7Error("Found remaining index of type %s in the final expression:\n%s\n Started with:\n%s" % (
                idx.to_canonical_string(),
                t.to_canonical_string(),
                t_input.to_canonical_string()
            ))

        if serialized_io:
            return (term_id, t.to_canonical_string())
        else:
            return (term_id, t)

    def build_matrix_element_expression_only_dots(self, process: HardCodedProcess, expressions: dict[Any, Any], mode: EvalMode) -> Any:

        match mode:
            case EvalMode.TREExTREE:
                me_expr_terms = self.sum_square_left_right_only_dots(
                    expressions["tree"]["graph_expressions"], expressions["tree"]["graph_expressions"])
                me_expr_terms = [t * sc.E(process.overall_factor)
                                 for t in me_expr_terms]
                return me_expr_terms

            case EvalMode.LOOPxTREE:
                raise MadLoop7Error(
                    "Loop x tree matrix element expression building not implemented yet!")
            case EvalMode.LOOPxLOOP:
                raise MadLoop7Error(
                    "Loop x loop matrix element expression building not implemented yet!")

    def export_evaluator_only_dots(self, process: HardCodedProcess, matrix_element_terms: list[Any], mode: EvalMode) -> None:

        madloop7_output = os.path.abspath(pjoin(self.config["output_path"], MADLOOP7_OUTPUT_DIR, process.name))  # nopep8
        if not os.path.isdir(madloop7_output):
            os.makedirs(madloop7_output)

        with open(pjoin(madloop7_output, "config.txt"), "w") as f:
            f.write(pformat(self.config))

        dim = sc.S("python::dim")
        mink = sc.tensors.Representation.mink(dim)
        P = sc.S("spenso::P")
        N = sc.S("spenso::N")
        dot = sc.S("spenso::dot", is_linear=True, is_symmetric=True)
        match mode:
            case EvalMode.TREExTREE:

                # Overall energy-momentum conservation
                # complement = sc.E("0")
                # complement_n = sc.E("0")
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
                        matrix_element_terms = [t.replace(
                            dot(N(i), P(j)), sc.E(f"dot_n{i+1}_{j+1}"), repeat=True) for t in matrix_element_terms]
                        if i <= j:
                            if i == j:
                                # Only support massless
                                dot_param = sc.E("0")
                            else:
                                dot_param = sc.E(f"dot_{i+1}_{j+1}")
                            matrix_element_terms = [t.replace(
                                dot(P(i), P(j)), dot_param, repeat=True) for t in matrix_element_terms]
                            matrix_element_terms = [t.replace(
                                dot(N(i), N(j)), sc.E(f"dot_n{i+1}_n{j+1}"), repeat=True) for t in matrix_element_terms]

                model_param_symbols = list(set(sum([[s for s in t.get_all_symbols(
                    False) if not str(s).startswith("dot")] for t in matrix_element_terms], [])))
                model = get_model(process.model)
                model_parameters = model.get('parameter_dict')
                matrix_element_terms = [t.expand().collect_factors()
                                        for t in matrix_element_terms]

                with open(pjoin(madloop7_output, "me_expression_terms.txt"), "w") as f:
                    f.write('[\n'+'\n,'.join(['"(%s)"' % t.to_canonical_string()
                            for t in matrix_element_terms])+'\n]')
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
                replace_dict['inline_asm'] = '"%s"' % self.config["inline_asm"]
                replace_dict['optimisation_level'] = self.config["optimisation_level"]
                replace_dict['targets'] = self.config["targets"]
                replace_dict['expand_before_building_evaluator'] = self.config["expand_before_building_evaluator"]

                model_param_values = []
                for p in model_param_symbols:
                    p_name = str(p).replace('spenso::', '')
                    if p_name not in model_parameters:
                        p_name = f'mdl_{p_name}'
                    model_param_values.append(
                        (p, model_parameters[p_name].value)
                    )
                replace_dict['model_parameters'] = ',\n'.join(
                    [f"(sc.S(\"{p.to_canonical_string()}\"), {v})" for (p, v) in model_param_values])

                with open(pjoin(madloop7_output, "run.py"), "w") as f:
                    f.write(run_sa_template % replace_dict)

            case EvalMode.LOOPxTREE:
                raise MadLoop7Error(
                    "Loop x tree matrix element expression evaluation not implemented yet!")
            case EvalMode.LOOPxLOOP:
                raise MadLoop7Error(
                    "Loop x loop matrix element expression evaluation not implemented yet!")

    @ staticmethod
    def expand_color(expression) -> list[tuple[Any, Any]]:

        f_, x_, x___, y__, z___ = sc.S("f_", "x_", "x___", "y__", "z___")
        color, coad, cof, dind = sc.S(
            "color", "spenso::coad", "spenso::cof", "spenso::dind")
        tmp = expression.replace_multiple([
            sc.Replacement(f_(x___, coad(y__), z___),
                           color(f_(x___, coad(y__), z___))),
            sc.Replacement(f_(x___, cof(y__), z___),
                           color(f_(x___, cof(y__), z___))),
            sc.Replacement(f_(x___, dind(cof(y__)), z___),
                           color(f_(x___, dind(cof(y__)), z___))),
        ])
        color_structures = [m[x_] for m in tmp.match(color(x_))]
        tmp = tmp.replace(color(x_), x_)
        return tmp.coefficient_list(*color_structures)

    @ staticmethod
    def expand_pol_vectors(expression) -> list[tuple[Any, Any]]:

        x_, x___ = sc.S("x_", "x___")
        polarization_vec = sc.S("polarization_vec")
        u, ubar, v, vbar, eps, epsbar = sc.S(
            "spenso::u", "spenso::ubar", "spenso::v", "spenso::vbar", "spenso::ϵ", "spenso::ϵbar")

        tmp = expression.replace_multiple([
            sc.Replacement(u(x___), polarization_vec(u(x___))),
            sc.Replacement(ubar(x___), polarization_vec(ubar(x___))),
            sc.Replacement(v(x___), polarization_vec(v(x___))),
            sc.Replacement(vbar(x___), polarization_vec(vbar(x___))),
            sc.Replacement(eps(x___), polarization_vec(eps(x___))),
            sc.Replacement(epsbar(x___), polarization_vec(epsbar(x___))),
        ])
        polarization_vec_structures = [m[x_]
                                       for m in tmp.match(polarization_vec(x_))]
        tmp = tmp.replace(polarization_vec(x_), x_)

        return tmp.coefficient_list(*polarization_vec_structures)

    def sum_square_left_right_tensor_networks(self, process: HardCodedProcess, left: dict[Any, Any], right: dict[Any, Any]) -> Any:
        """Combine left and right expressions into a single expression."""

        dim = sc.S("python::dim")
        mink = sc.tensors.Representation.mink(dim)
        bis = sc.tensors.Representation.bis(dim)
        mink4 = sc.tensors.Representation.mink(4)
        Q_symbol = sc.S("spenso::Q")
        P_symbol = sc.S("spenso::P")
        N_symbol = sc.S("spenso::N")
        Q = sc.tensors.TensorStructure(mink, name=Q_symbol)
        P = sc.tensors.TensorStructure(mink, name=P_symbol)
        N = sc.tensors.TensorStructure(mink, name=N_symbol)
        Q4 = sc.tensors.TensorStructure(mink4, name=Q_symbol)
        P4 = sc.tensors.TensorStructure(mink4, name=P_symbol)
        N4 = sc.tensors.TensorStructure(mink4, name=N_symbol)

        def p(i, j):
            return P(i, ';', j)

        def n(i, j):
            return N(i, ';', j)

        def p4(i, j):
            return P4(i, ';', j)

        def n4(i, j):
            return N4(i, ';', j)

        my_lib = sc.tensors.TensorLibrary.hep_lib()

        l_side = sc.S("l")
        r_side = sc.S("r")

        gamma = sc.tensors.TensorStructure(
            mink, bis, bis, name=sc.S('spenso::gamma'))

        g = sc.tensors.TensorStructure(mink, mink, name=sc.S('spenso::g'))

        bis = sc.tensors.Representation.bis(4)
        u, ubar, v, vbar, eps, epsbar = sc.S(
            "spenso::u", "spenso::ubar", "spenso::v", "spenso::vbar", "spenso::ϵ", "spenso::ϵbar")
        i_, j_, d_, a_, b_ = sc.S("i_", "j_", "d_", "a_", "b_")
        dummy = sc.S("dummy")
        dummy_ss = sc.S("dummy_ss")

        dot = sc.S("spenso::dot", is_linear=True, is_symmetric=True)
        overall_emr_replacements = {}

        logger.info("Loading graph expressions%s..." % (
            (" and gamma simplifying amplitudes" if self.config['simplify_amplitude_gamma_algebra'] else "")
        ))

        expressions = {'left': left, 'right': right}
        lorentz_tensors = {}
        polarization_vecs = {'left': None, 'right': None}
        polarization_vec_pieces = {
            i_ext: {'left': None, 'right': None} for i_ext in range(process.n_external)}
        for i_side, (key, wrapper) in enumerate(zip(['left', 'right'], [l_side, r_side])):
            side_expression = sc.E("0")
            for graph_id, graph_expr in sorted(expressions[key].items(), key=lambda x: x[0]):
                num, denom, emr_to_lmb_replacements = self.build_graph_expression(
                    graph_id, graph_expr, to_lmb=(not self.config['work_in_emr']))
                overall_emr_replacements.update(emr_to_lmb_replacements)

                if key == 'right':
                    num = sc.algebraic_simplification.conj(num)
                    denom = sc.algebraic_simplification.conj(denom)
                num = sc.algebraic_simplification.wrap_dummies(num, wrapper)

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

                pattern = sc.E("f_")(sc.E("x___"), sc.E("rep_")
                                     (sc.E("y___"), wrapper(sc.E("x_"))))
                for m in pol_vectors.match(pattern):
                    # a_match = pattern.replace_multiple(
                    #     [sc.Replacement(k, v, k.req_lit()) for k, v in m.items()], repeat=True)
                    a_match = pattern.to_canonical_string()
                    for k, e in m.items():
                        a_match = re.sub(k.to_canonical_string(),
                                         e.to_canonical_string(), a_match)
                    a_match = sc.E(a_match)
                    ext_id = int(m[sc.S('x_')].to_canonical_string())
                    if polarization_vec_pieces[ext_id][key] is None:
                        polarization_vec_pieces[ext_id][key] = a_match
                    else:
                        if polarization_vec_pieces[ext_id][key] != a_match:
                            raise MadLoop7Error(
                                f"Polarization vector pieces for {key} external #{ext_id} do not match between graphs")
                color_lorentz_splits = self.expand_color(num)
                for lor_struct_id, (col, lor) in enumerate(color_lorentz_splits):
                    # print("lor ", lor)
                    # print("")
                    # print("lor simp ", sc.algebraic_simplification.simplify_gamma(lor))
                    # print("")
                    # print("col ", col)
                    # print("")
                    # print("col simp", sc.algebraic_simplification.simplify_gamma(col))
                    # stop
                    lor = sc.algebraic_simplification.simplify_metrics(lor)
                    if self.config['simplify_amplitude_gamma_algebra']:
                        # print("BEFORE GAMMA SIMPLIFICATION\n%s" % lor.to_canonical_string())  # nopep8
                        lor = sc.algebraic_simplification.simplify_gamma(lor)
                        # print("AFTER GAMMA SIMPLIFICATION\n%s" % lor.to_canonical_string())  # nopep8
                    lor = lor.collect_num()
                    # col = sc.algebraic_simplification.simplify_color(col)
                    lorentz_tensors[(i_side, graph_id, lor_struct_id)] = lor  # nopep8
                    side_expression = side_expression + \
                        sc.S("color")(col) * \
                        sc.S("denom")(denom) * \
                        sc.S("T")(i_side, graph_id, lor_struct_id)

            expressions[key] = side_expression

        assert (expressions['left'] is not None)
        assert (expressions['right'] is not None)

        if polarization_vecs['left'] is None:
            polarization_vecs['left'] = sc.E("1")
        if polarization_vecs['right'] is None:
            polarization_vecs['right'] = sc.E("1")
        assert (polarization_vecs['left'] is not None)
        assert (polarization_vecs['right'] is not None)

        for k, e in polarization_vec_pieces.items():
            if e['left'] is None:
                e['left'] = sc.E("1")
            if e['right'] is None:
                e['right'] = sc.E("1")
            assert (e['left'] is not None)
            assert (e['right'] is not None)

        # print("polarization_vecs['left']=", polarization_vecs['left'])
        # print("polarization_vecs['right']=", polarization_vecs['right'])
        # pprint(polarization_vec_pieces)

        pol_spin_sum_input = (
            polarization_vecs['left'] * polarization_vecs['right'])
        pol_spin_sum_input_pieces = {k: e['left']*e['right'] for k, e in polarization_vec_pieces.items()}  # type: ignore # nopep8
        # For now only support massless externals
        if self.config['physical_vector_polarization_sum']:
            transverse_physical_vector_sum_num = n(i_, dummy_ss(i_, 1)) * p(i_, dummy_ss(i_, 1)) * (
                p(i_, l_side(a_)) * n(i_, r_side(a_)) +
                n(i_, l_side(a_)) * p(i_, r_side(a_))
            )
            if not GAUGE_VECTOR_SQUARED_IS_ZERO:
                transverse_physical_vector_sum_num = transverse_physical_vector_sum_num - \
                    (
                        n(i_, dummy_ss(i_, 2)) * n(i_, dummy_ss(i_, 2)) *
                        p(i_, l_side(a_)) * p(i_, r_side(a_))
                    )
            transverse_physical_vector_sum_denom_a = n(i_, dummy_ss(i_, 3)) * p(i_, dummy_ss(i_, 3)) * n(i_, dummy_ss(i_, 4)) * p(i_, dummy_ss(i_, 4))  # nopep8
            transverse_physical_vector_sum_denom_b = dot(N_symbol(i_), P_symbol(i_))**2  # nopep8
        else:
            transverse_physical_vector_sum_num = sc.E("0")
            transverse_physical_vector_sum_denom_a = sc.E("1")
            transverse_physical_vector_sum_denom_b = sc.E("1")

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
                        sc.E("1")
                    )
                ),
                (
                    ubar(i_, bis(sides[0](a_)))*u(j_, bis(sides[1](a_))),
                    (
                        -gamma(sides[0](a_), sides[1](a_), dummy(i_, a_)) * p(i_, dummy(i_, a_)),  # nopep8
                        sc.E("1")
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

        pol_spin_sum_input_pieces_num = {}
        pol_spin_sum_input_pieces_denom = {}
        for k, e in pol_spin_sum_input_pieces.items():
            pol_spin_sum_input_pieces_num[k] = e
            pol_spin_sum_input_pieces_denom[k] = e
            for src, (trgt_num, trgt_denom) in spin_sum_rules:
                pol_spin_sum_input_pieces_num[k] = pol_spin_sum_input_pieces_num[k].replace(
                    src, trgt_num, repeat=True)
                pol_spin_sum_input_pieces_denom[k] = pol_spin_sum_input_pieces_denom[k].replace(
                    src, trgt_denom, repeat=True)

        # Define choice of gauge vectors
        # "temporal": n = (1,0,0,0)
        # "k-axial": n = (k^0, -k_vec)
        # GAUGE_VECTOR_CHOICE = "madgraph"
        GAUGE_VECTOR_CHOICE = "k-axial"

        for i in range(process.n_external):
            n_temporal_vector = sc.tensors.LibraryTensor.sparse(
                sc.tensors.TensorStructure(mink4, sc.E(f"{i}"), name=N_symbol), type(sc.E("1")))
            match GAUGE_VECTOR_CHOICE:
                case "temporal":
                    n_temporal_vector[[0,]] = sc.E("1")
                case "k-axial":
                    # t = sc.tensors.TensorNetwork(p4(i, 1)*sc.tensors.TensorStructure.id(mink4)(1, 2))
                    # print(dir(sc.tensors.TensorName))
                    # print(sc.tensors.TensorName.flat)
                    # print(sc.tensors.TensorName.flat(mink4, mink4, 1, 2))
                    # sc.tensors.TensorStructure(mink4, mink4, name=id_tensor)(1, 2))
                    # stop
                    t = sc.tensors.TensorNetwork(p4(i, 1))*sc.tensors.TensorNetwork(sc.tensors.TensorName.flat(mink4(1), mink4(2)))  # nopep8
                    t.execute(my_lib)
                    t = t.result_tensor(my_lib)
                    for t_idx in t.structure():
                        n_temporal_vector[t_idx] = t[t_idx]
            my_lib.register(n_temporal_vector)
        # Process the tensor networks
        for k in pol_spin_sum_input_pieces_num:
            e = sc.algebraic_simplification.cook_indices(
                pol_spin_sum_input_pieces_num[k]).replace(dim, sc.E("4"))

            t = sc.tensors.TensorNetwork(e, my_lib)
            t.execute(my_lib)
            # print(t)
            t = t.result_tensor(my_lib)
            # print(t)
            t2 = t
            for i in t.structure():
                try:
                    t2[i] = t[i].replace(sc.E(f"P({k},cind(0))^2", "spenso"), sc.E(
                        f"(P({k},cind(1))^2+P({k},cind(2))^2+P({k},cind(3))^2)", "spenso")).expand()
                except IndexError as e:
                    t2[i] = sc.E("0")

            # pol_spin_sum_input_pieces_num[k] = t
            net = sc.tensors.TensorNetwork.one()*t2

            net.execute()
            pol_spin_sum_input_pieces_num[k] = net

        # pprint(pol_spin_sum_input_pieces_num)

        # Add dot products replacement
        # dot_products_replacement = []
        # for i in range(process.n_external):
        #     for j in range(process.n_external):
        #         if i == j:
        #             dot_products_replacement.append((
        #                 dot(P_symbol(i), P_symbol(j)),
        #                 sc.E("0")
        #             ))
        #         elif i < j:
        #             t = sc.tensors.TensorNetwork(
        #                 p4(i, 1)*p4(j, 1), my_lib)
        #             t.execute(my_lib)
        #             t = t.result_tensor(my_lib)
        #             dot_products_replacement.append((dot(P_symbol(i), P_symbol(j)), t[[]]))  # nopep8

        #         t = sc.tensors.TensorNetwork(n4(i, 1)*n4(j, 1), my_lib)
        #         t.execute(my_lib)
        #         t = t.result_tensor(my_lib)
        #         dot_products_replacement.append((dot(N_symbol(i), N_symbol(j)), t[[]]))  # nopep8

        #         t = sc.tensors.TensorNetwork(n4(i, 1)*p4(j, 1), my_lib)
        #         t.execute(my_lib)
        #         t = t.result_tensor(my_lib)

        #         dot_products_replacement.append((dot(N_symbol(i), P_symbol(j)), t[[]]))  # nopep8

        #         t = sc.tensors.TensorNetwork(p4(i, 1)*n4(j, 1), my_lib)
        #         t.execute(my_lib)
        #         t = t.result_tensor(my_lib)
        #         dot_products_replacement.append((dot(P_symbol(i), N_symbol(j)), t[[]]))  # nopep8

        me = expressions['left'] * expressions['right']  # type: ignore

        # Unwrap color and simplify
        me = me.replace(sc.E("color(x__)"), sc.E("x__"))
        # me = me.replace(sc.E("denom(x__)"), sc.E("1"))
        # me = me.replace(sc.E("T(x_,y_,z_)"), sc.E("1"))
        # print("Before color simplification:", me)
        me = sc.algebraic_simplification.simplify_color(me)
        me = MadLoop7.substitute_constants(me)

        # Merge identical emr vectors across graphs
        unique_emrs = {}
        emrs_to_merge = {}
        for emr_v, lmb_expr in overall_emr_replacements.items():
            if lmb_expr not in emrs_to_merge:
                unique_emrs[emr_v] = lmb_expr
                emrs_to_merge[lmb_expr] = [emr_v,]
            else:
                emrs_to_merge[lmb_expr].append(emr_v)

        emr_to_lmb_replacements = unique_emrs

        emr_merging_replacements = []
        for emr in emrs_to_merge.values():
            emr_merging_replacements.extend([sc.Replacement(emr_v, emr[0]) for emr_v in emr[1:]])  # type: ignore # nopep8
        if len(emr_merging_replacements) > 0:
            logger.info("Merging %d EMR vectors" %
                        len(emr_merging_replacements))
            me = me.replace_multiple(emr_merging_replacements)

        # Now address Lorentz
        status_line = "Doing T%s*T%s : %-60s | expr_size: %-8d bytes"

        def process_lorentz(match):
            left_key = (0, int(str(match[sc.S("LeftGraphID_")])), int(str(match[sc.S("LeftTermID_")])))  # nopep8
            right_key = (1, int(str(match[sc.S("RightGraphID_")])), int(str(match[sc.S("RightTermID_")])))  # nopep8
            # VHHACK
            # if left_key[1] == right_key[1]:
            #     return sc.E("0")
            print(status_line % (left_key, right_key, "start", -1), end=END_CHAR)  # nopep8
            left_tensor = lorentz_tensors[left_key]
            # print("\n\n left_tensor=\n", left_tensor)
            right_tensor = lorentz_tensors[right_key]
            # print("\n\n right_tensor=\n", right_tensor)
            tensor_expr = left_tensor * right_tensor

            # my_lib = sc.tensors.TensorLibrary.hep_lib()

            # tensor_expr = tensor_expr.replace(sc.E("spenso::gamma(spenso::mink(x__),y__)"), sc.E(
            #     "spenso::gamma(y__,spenso::mink(x__))"))

            # HACK
            # HACK_MODE_I = int(str(match[sc.S("LeftGraphID_")])) + \
            #     int(str(match[sc.S("LeftTermID_")]))
            # tensor_expr = p4((HACK_MODE_I) % 3, 1)*p4((HACK_MODE_I+1) % 3, 1)
            tensor_expr = sc.algebraic_simplification.cook_indices(
                tensor_expr).replace(dim, sc.E("4"))

            # print(tensor_expr)
            # print(tensor_expr.to_canonical_string())
            # tensor_expr = sc.E("""(
            #                    -1*    spenso::g(spenso::mink(4,python::l_6),spenso::mink(4,python::l_9))*spenso::g(spenso::mink(4,python::l_7),spenso::mink(4,python::l_8))
            #                    +      spenso::g(spenso::mink(4,python::l_6),spenso::mink(4,python::l_8))*spenso::g(spenso::mink(4,python::l_7),spenso::mink(4,python::l_9)))
            #                    *-1𝑖 * (spenso::G^3
            #                         * spenso::g(spenso::bis(4,python::l_2),spenso::bis(4,python::l_5))
            #                         * spenso::g(spenso::bis(4,python::l_3),spenso::bis(4,python::l_6))
            #                         * spenso::g(spenso::mink(4,python::l_0),spenso::mink(4,python::l_6))
            #                         * spenso::g(spenso::mink(4,python::l_1),spenso::mink(4,python::l_7))
            #                         * spenso::g(spenso::mink(4,python::l_4),spenso::mink(4,python::l_8))
            #                         * spenso::g(spenso::mink(4,python::l_5),spenso::mink(4,python::l_9))
            #                         * spenso::gamma(spenso::bis(4,python::l_6),spenso::bis(4,python::l_5),spenso::mink(4,python::l_5))
            # """)

            # print("\n\ninput: \n%s\n\n" % (tensor_expr.to_canonical_string()))
            t_size = tensor_expr.get_byte_size()
            print(status_line % (left_key, right_key, "Parsing tensor network...", t_size), end=END_CHAR)  # nopep8
            my_lib = sc.tensors.TensorLibrary.hep_lib()

            t = sc.tensors.TensorNetwork(tensor_expr, my_lib)
            # print(str(t))
            # print(len(str(t)))

            print(status_line % (left_key, right_key, "Multiplying in polarization sum...", t_size), end=END_CHAR)  # nopep8

            # tensor_expr = tensor_expr * pol_spin_sum_num
            for v in pol_spin_sum_input_pieces_num.values():
                t = t * v

            # print("\n\nsc.Tensor network:\n%s\n\n", t)
            if not self.config['step_through_network_execution']:
                print(status_line % (left_key, right_key, "Executing tensor network of size %-10s..." % (len(str(t))), t_size), end=END_CHAR)  # nopep8
                t.execute(my_lib)
            else:
                i_step = 0
                while True:
                    i_step += 1
                    print(status_line % (left_key, right_key, "Executing scalar tensor network reduction, step #%d with size %-10s..." % (i_step, len(str(t))), t_size), end=END_CHAR)  # nopep8
                    # print(str(t))
                    t.execute(my_lib, 1, sc.tensors.ExecutionMode.Scalar)
                    # print("DONE")

                    i_step += 1
                    print(status_line % (left_key, right_key, "Executing single tensor network reduction, step #%d with size %-10s..." % (i_step, len(str(t))), t_size), end=END_CHAR)  # nopep8
                    # print(str(t))
                    t.execute(my_lib, 1, sc.tensors.ExecutionMode.Single)
                    # print("DONE")
                    try:
                        _ = t.result_tensor(my_lib)
                    except:
                        continue
                    break

            print(status_line % (left_key, right_key, "Fetching result from tensor network...", t_size), end=END_CHAR)  # nopep8
            # t = t.result_scalar(my_lib)
            t = t.result_tensor(my_lib)

            # print(t)
            res = t[[]]

            # print("Resulting tensor:", res.to_canonical_string())
            # print("Resulting tensor:", res)
            # print("DONE!")

            return res

        me = me.replace(sc.E("T(0,LeftGraphID_,LeftTermID_)*T(1,RightGraphID_,RightTermID_)"),
                        lambda m: process_lorentz(m))

        # print("Before Dot product substitution!\n", me)

        # Multiply in external polarization vectors spin-sum
        # me = me * sc.E("denom")(pol_spin_sum_denom)
        for v in pol_spin_sum_input_pieces_denom.values():
            me = me * sc.E("denom")(v)

        def process_dot(match):
            dot_a, dot_b = match[sc.S("a_")], match[sc.S("b_")]
            spenso_dot = (dot_a*dot_b).replace(sc.E("v_(id_)"),
                                               sc.E("v_(id_,spenso::mink(4,1))"))
            t = sc.tensors.TensorNetwork(spenso_dot, my_lib)
            t.execute(my_lib)
            t = t.result_tensor(my_lib)
            return t[[]]

        # Substitute dot products appearing in denominators now
        def process_denom(match):
            denom_expr = match[sc.S("x_")]
            # print("Processing denom:", denom_expr.to_canonical_string())
            # for src, trgt in dot_products_replacement:
            #     denom_expr = denom_expr.replace(src, trgt, repeat=True)
            # print("Processed denom:", denom_expr.to_canonical_string())
            res = sc.E("denom")(denom_expr.replace(
                sc.E("spenso::dot(a_,b_)"), lambda m: process_dot(m)))
            return res

        me = me.replace(sc.E("denom(x_)"), lambda m: process_denom(m))

        # Unwrap denominators
        me = me.replace(sc.E("denom(x_)"), sc.E("1/x_"))

        # print("Final result:\n", me)
        # print("Final len:\n", len(me.expand()))
        # stop
        return me, overall_emr_replacements

    # @staticmethod
    # def FUFU():
    #     import sys
    #     sys.path.insert(
    #         0, "/Users/vjhirsch/HEP_programs/symbolica-community/python")
    #     from symbolica_community import Expression, S, E
    #     from symbolica_community.tensors import TensorName as N, LibraryTensor, TensorNetwork, Representation, TensorStructure, TensorIndices, Tensor, Slot, TensorLibrary
    #     import symbolica_community
    #     import symbolica_community.tensors as tensors
    #     import random
    #     import time
    #     tensor_input_A = """
    #     -1*spenso::g(spenso::mink(4,python::l_6),spenso::mink(4,python::l_9))
    #     *spenso::g(spenso::mink(4,python::l_7),spenso::mink(4,python::l_8))
    #     """.replace('\n', '').split("*")
    #     tensor_input_B = """
    #     spenso::g(spenso::mink(4,python::l_6),spenso::mink(4,python::l_8))
    #     *spenso::g(spenso::mink(4,python::l_7),spenso::mink(4,python::l_9))
    #     *-1𝑖
    #     *spenso::G^3*spenso::g(spenso::bis(4,python::l_2),spenso::bis(4,python::l_5))
    #     *spenso::g(spenso::bis(4,python::l_3),spenso::bis(4,python::l_6))
    #     *spenso::g(spenso::mink(4,python::l_0),spenso::mink(4,python::l_6))
    #     *spenso::g(spenso::mink(4,python::l_1),spenso::mink(4,python::l_7))
    #     *spenso::g(spenso::mink(4,python::l_4),spenso::mink(4,python::l_8))
    #     *spenso::g(spenso::mink(4,python::l_5),spenso::mink(4,python::l_9))
    #     *spenso::gamma(spenso::bis(4,python::l_6),spenso::bis(4,python::l_5),spenso::mink(4,python::l_5))
    #     """.replace('\n', '').split("*")
    #     import random
    #     random.shuffle(tensor_input_A)
    #     random.shuffle(tensor_input_B)
    #     tensor_input = "(%s)+(%s)" % ("*".join(tensor_input_A),
    #                                   "*".join(tensor_input_B))
    #     print(tensor_input)
    #     all_symbols = ['spenso::g', 'spenso::gamma', 'spenso::mink', 'spenso::bis', 'python::l_6', 'python::l_9', 'python::l_7',
    #                    'python::l_8', 'python::l_2', 'python::l_5', 'python::l_1', 'python::l_4', 'spenso::G', 'python::l_0', 'python::l_3']
    #     random.shuffle(all_symbols)
    #     print(all_symbols)
    #     # for s in all_symbols:
    #     #    _ = S("s")
    #     # tensor_input = """
    #     # (spenso::g(spenso::mink(4,python::l_0),spenso::mink(4,python::l_6))*spenso::gamma(spenso::bis(4,python::l_6),spenso::bis(4,python::l_5),spenso::mink(4,python::l_5))*spenso::g(spenso::bis(4,python::l_2),spenso::bis(4,python::l_5))*spenso::g(spenso::mink(4,python::l_6),spenso::mink(4,python::l_8))*spenso::g(spenso::mink(4,python::l_5),spenso::mink(4,python::l_9))*spenso::g(spenso::mink(4,python::l_4),spenso::mink(4,python::l_8))*spenso::g(spenso::mink(4,python::l_1),spenso::mink(4,python::l_7))*spenso::G^3*spenso::g(spenso::bis(4,python::l_3),spenso::bis(4,python::l_6))*-1𝑖*spenso::g(spenso::mink(4,python::l_7),spenso::mink(4,python::l_9)))+(-1*spenso::g(spenso::mink(4,python::l_7),spenso::mink(4,python::l_8))*spenso::g(spenso::mink(4,python::l_6),spenso::mink(4,python::l_9)))
    #     # """
    #     tensor_input = """
    #     -1𝑖*G^3*(g(mink(4,l_6),mink(4,l_8))*g(mink(4,l_7),mink(4,l_9))-g(mink(4,l_6),mink(4,l_9))*g(mink(4,l_8),mink(4,l_7)))*g(mink(4,l_0),mink(4,l_6))*g(mink(4,l_1),mink(4,l_7))*g(mink(4,l_4),mink(4,l_8))*g(mink(4,l_5),mink(4,l_9))*g(bis(4,l_2),bis(4,l_5))*g(bis(4,l_3),bis(4,l_6))*gamma(bis(4,l_6),bis(4,l_5),mink(4,l_5))
    #     """
    #     for s in all_symbols:
    #         namespace, v = s.split('::')
    #         print(namespace, v)
    #         tensor_input = tensor_input.replace(v, s)
    #         tensor_input = tensor_input.replace("spenso::spenso", "spenso")
    #         tensor_input = tensor_input.replace("python::python", "python")

    #     print(tensor_input)
    #     tensor_expr = E(tensor_input)
    #     # print([v.to_canonical_string() for v in tensor_expr.get_all_symbols()])
    #     hep_lib = TensorLibrary.hep_lib()
    #     t = TensorNetwork(tensor_expr, hep_lib)
    #     # print(len(str(t)))
    #     print("DOING IT")
    #     # print(str(t))
    #     if True:
    #         if False:
    #             t.execute(hep_lib)
    #         else:
    #             i_step = 0
    #             while True:
    #                 time.sleep(0.01)
    #                 i_step += 1
    #                 print(i_step, len(str(t)))
    #                 print(str(t))
    #                 t.execute(hep_lib, 1, tensors.ExecutionMode.Scalar)

    #                 i_step += 1
    #                 print(i_step, len(str(t)))
    #                 print(str(t))
    #                 t.execute(hep_lib, 1, tensors.ExecutionMode.Single)
    #                 if i_step > 20:
    #                     break
    #                 try:
    #                     _ = t.result_tensor(my_lib)
    #                 except:
    #                     continue
    #                 break
    #         # len(t.result_tensor())
    #         # print(str(t))
    #         print("DONE ", len(str(t)))

    def build_matrix_element_expression_tensor_networks(self, process: HardCodedProcess, expressions: dict[Any, Any], mode: EvalMode) -> Any:

        match mode:
            case EvalMode.TREExTREE:
                me_expr, overall_emr_replacements = self.sum_square_left_right_tensor_networks(
                    process, expressions["tree"]["graph_expressions"], expressions["tree"]["graph_expressions"])
                me_expr = me_expr * sc.E(process.overall_factor)
                return me_expr, overall_emr_replacements

            case EvalMode.LOOPxTREE:
                raise MadLoop7Error(
                    "Loop x tree matrix element expression building not implemented yet!")
            case EvalMode.LOOPxLOOP:
                raise MadLoop7Error(
                    "Loop x loop matrix element expression building not implemented yet!")

    def export_evaluator_tensor_networks(self, process: HardCodedProcess, matrix_element_expression: Any, overall_emr_replacements: dict[Any, Any], mode: EvalMode) -> None:
        logger.info(
            "Exporting evaluator tensor networks for process %s in mode %s", process.name, mode)

        madloop7_output = os.path.abspath(pjoin(self.config["output_path"], MADLOOP7_OUTPUT_DIR, process.name))  # nopep8
        if not os.path.isdir(madloop7_output):
            os.makedirs(madloop7_output)

        with open(pjoin(madloop7_output, "config.txt"), "w") as f:
            f.write(pformat(self.config))

        mink4 = sc.tensors.Representation.mink(4)
        P_symbol = sc.S("spenso::P")
        Q_symbol = sc.S("spenso::Q")
        EMRID_symbol = sc.S("spenso::EMRID")
        P4 = sc.tensors.TensorStructure(mink4, name=P_symbol)
        Q4 = sc.tensors.TensorStructure(mink4, name=Q_symbol)

        def p4(i, j):
            return P4(i, ';', j)

        def q4(graph_id, edge_id, j):
            return Q4(EMRID_symbol(graph_id, edge_id), ';', j)

        my_lib = sc.tensors.TensorLibrary.hep_lib()
        match mode:
            case EvalMode.TREExTREE:

                # Overall energy-momentum conservation
                # complement = sc.E("0")
                # complement_n = sc.E("0")
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
                logger.info("Computing external momenta replacements...")
                kinematic_symbols = []
                external_replacements = []
                me_replacements = []
                for i in range(process.n_external):
                    t = sc.tensors.TensorNetwork(p4(i, 1))
                    t.execute(my_lib)
                    t = t.result_tensor(my_lib)
                    for i_idx, lor_idx in enumerate(t):
                        kinematic_symbols.append(sc.E(f"kin_p{i+1}_{i_idx}"))
                        external_replacements.append(
                            sc.Replacement(lor_idx, kinematic_symbols[-1]))
                        me_replacements.append(sc.Replacement(
                            lor_idx, kinematic_symbols[-1]))

                logger.info("Computing EMR momenta replacements...")
                functions = []
                for q, ps in overall_emr_replacements.items():
                    graph_id, edge_id = None, None
                    for m in q.match(sc.E("spenso::Q(spenso::EMRID(i_,j_),cind_)")):
                        graph_id = int(m[sc.S("i_")].to_canonical_string())
                        edge_id = int(m[sc.S("j_")].to_canonical_string())
                    t = sc.tensors.TensorNetwork(q4(graph_id, edge_id, 1))
                    t.execute(my_lib)
                    t = t.result_tensor(my_lib)
                    for i_idx, lor_idx in enumerate(t):
                        lmb_expr = lor_idx.replace(
                            q, ps).replace_multiple(external_replacements)
                        emr_symbol = sc.S(
                            f"kin_q{graph_id+1}_{edge_id+1}_{i_idx}")  # type: ignore # nopep8
                        functions.append(
                            (
                                emr_symbol,
                                lmb_expr
                            )
                        )
                        me_replacements.append(
                            sc.Replacement(lor_idx, emr_symbol()))

                logger.info("Applying replacements...")
                matrix_element_expression = matrix_element_expression.replace_multiple(
                    me_replacements)

                logger.info("Identifying non kinematic parameters...")

                # Do a cleaner filter eventually,
                model_param_symbols = [s for s in matrix_element_expression.get_all_symbols(
                    False) if s not in kinematic_symbols]

                model = get_model(process.model)
                model_parameters = model.get('parameter_dict')
                matrix_element_expression = matrix_element_expression.collect_factors()

                logger.info("Exporting standalone resources...")

                with open(pjoin(madloop7_output, "me_expression.txt"), "w") as f:
                    f.write(matrix_element_expression.to_canonical_string())

                with open(pjoin(madloop7_output, "me_functions.txt"), "w") as f:
                    f.write(str([(f[0].to_canonical_string(), f[1].to_canonical_string())
                            for f in sorted(functions)]))

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
                replace_dict['inline_asm'] = '"%s"' % self.config["inline_asm"]
                replace_dict['optimisation_level'] = self.config["optimisation_level"]
                replace_dict['targets'] = self.config["targets"]
                replace_dict['expand_before_building_evaluator'] = self.config["expand_before_building_evaluator"]

                model_param_values = []
                for p in model_param_symbols:
                    p_name = str(p).replace('spenso::', '')
                    if p_name not in model_parameters:
                        p_name = f'mdl_{p_name}'
                    model_param_values.append(
                        (p, model_parameters[p_name].value)
                    )
                replace_dict['model_parameters'] = ',\n'.join(
                    [f"(sc.S(\"{p.to_canonical_string()}\"), {v})" for (p, v) in model_param_values])

                with open(pjoin(madloop7_output, "run.py"), "w") as f:
                    f.write(run_sa_template % replace_dict)

            case EvalMode.LOOPxTREE:
                raise MadLoop7Error(
                    "Loop x tree matrix element expression evaluation not implemented yet!")
            case EvalMode.LOOPxLOOP:
                raise MadLoop7Error(
                    "Loop x loop matrix element expression evaluation not implemented yet!")

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
            if os.path.isfile(pjoin(self.config["output_path"], GAMMALOOP_OUTPUT_DIR, f"{output_name}.gL")):
                os.remove(
                    pjoin(self.config["output_path"], GAMMALOOP_OUTPUT_DIR, f"{output_name}.gL"))
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
