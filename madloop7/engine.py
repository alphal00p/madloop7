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

root_path = os.path.abspath(os.path.dirname(__file__))

pjoin = os.path.join

MADSYMBOLIC_OUTPUT_DIR = "madsymbolic_outputs"
MADGRAPH_OUTPUT_DIR = "madgraph_outputs"
GAMMALOOP_OUTPUT_DIR = "gammaloop_outputs"
MADLOOP7_OUTPUT_DIR = "madloop7_outputs"


class EvalMode(Enum):
    """Evaluation modes for MadLoop7."""
    TREExTREE = "tree-level"
    LOOPxTREE = "virtual"
    LOOPxLOOP = "loop-induced"


class MadLoop7(object):

    def __init__(self, config_path: str) -> None:
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def generate_process_gamma_loop(self, process: HardCodedProcess) -> None:
        logger.critical("GammaLoop generation not implemented yet")
        pass

    def generate_process_madgraph(self, process: HardCodedProcess) -> None:
        assert process.madgraph_generation is not None
        assert process.madsymbolic_output is not None
        logger.info(f"Generating process {process.name} with MG5aMC...")  # nopep8
        process_card = os.path.abspath(pjoin(self.config["output_path"], MADGRAPH_OUTPUT_DIR, f"{process.name}.mg5"))  # nopep8
        yaml_graph_outputs = os.path.abspath(pjoin(self.config["output_path"], MADSYMBOLIC_OUTPUT_DIR, f"{process.name}"))  # nopep8
        madgraph_output = os.path.abspath(pjoin(self.config["output_path"], MADGRAPH_OUTPUT_DIR, process.name))  # nopep8
        graph_outputs = [
            (
                graph_class,
                os.path.abspath(pjoin(self.config["output_path"], MADSYMBOLIC_OUTPUT_DIR, process.name, ms_output)),  # nopep8
                os.path.abspath(pjoin(self.config["output_path"], GAMMALOOP_OUTPUT_DIR, f"{process.name}_{graph_class}.dot"))  # nopep8
             )  # nopep8
            for (graph_class, ms_output) in process.madsymbolic_output
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
                        f"gL import_graphs {madsymbolic_output} --format yaml",  # nopep8
                        f"gL export_graphs {dot_output}",  # nopep8
                    ])
                    for (_graph_class, madsymbolic_output, dot_output) in graph_outputs
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

        for (_graph_class, _madsymbolic_output, dot_output) in graph_outputs:
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

        try:
            from gammaloop.interface.gammaloop_interface import GammaLoop, CommandList  # type: ignore # nopep8
        except:
            gammaloop_path = pjoin(self.config["gammaloop_path"], "python")
            if gammaloop_path not in sys.path:
                sys.path.insert(0, gammaloop_path)
            try:
                from gammaloop.interface.gammaloop_interface import GammaLoop, CommandList  # type: ignore # nopep8
            except:
                raise MadLoop7Error('\n'.join([
                    "ERROR: Could not import Python's gammaloop module.",
                    "Add '<GAMMALOOP_INSTALLATION_DIRECTORY>/python' to your PYTHONPATH or specify it under 'gammaloop_path' in the configuration file used to load MadLoop7.",]))

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

    def build_evaluators_with_spenso(self, process: HardCodedProcess, tree_graph_ids: list[int] | None = None, loop_graph_ids: list[int] | None = None) -> None:
        logger.info("Build graph evaluators with spenso...")

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
                graph_id = int(graph_expr_path.split("_")[-2])
                graph_ids = tree_graph_ids if graph_class == "tree" else loop_graph_ids
                if graph_ids is None or graph_id in graph_ids:
                    with open(pjoin(self.config["output_path"], GAMMALOOP_OUTPUT_DIR, output_name, "sources", output_metadata["output_type"], output_metadata["contents"][0], "expressions", graph_expr_path), "r") as f:
                        expr = json.load(f)
                        graph_expressions[graph_id] = {
                            'expression': expr[0],
                            'momenta': dict(expr[1]),
                            'denominators': dict(expr[2]),
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

        matrix_element_expression = self.build_matrix_element_expression(
            process, expressions,  eval_mode)

        self.export_evaluator(process, matrix_element_expression, eval_mode)
        # tensor_nets = self.build_tensor_networks(expressions, eval_mode)

    def build_graph_expression(self, graph_expr: dict[Any, Any], to_lmb: bool = False) -> Any:
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

        numerator = curate(E_sp(graph_expr['expression']))
        denominator = E("1")
        for (p, m) in graph_expr['denominators'].items():
            denominator = denominator * \
                (S("symbolica_community::dot", is_linear=True,
                 is_symmetric=True)(E_sp(p), E_sp(p)) - E_sp(m)**2)

        e = numerator / denominator
        if to_lmb:
            for (src, trgt) in graph_expr['momenta'].items():
                e = e.replace(E_sp(src), E_sp(trgt), repeat=True)

        return e

    def sum_square_left_right(self, left: dict[Any, Any], right: dict[Any, Any]) -> Any:
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
        PHYSICAL_VECTOR_SUM_RULE = True

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
                wrap_dummies(self.build_graph_expression(
                    graph_expr, to_lmb=True), l_side)
                for _graph_id, graph_expr in sorted(expressions['left'].items(), key=lambda x: x[0])
            ]
            diagram_expressions['right'] = [
                conj(wrap_dummies(self.build_graph_expression(
                    graph_expr, to_lmb=True), r_side))
                for _graph_id, graph_expr in sorted(expressions['right'].items(), key=lambda x: x[0])
            ]
            for left_expr in diagram_expressions['left']:
                for right_expr in diagram_expressions['right']:
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

            t = simplify_color(t)
            t = simplify_gamma(t)
            t = to_dots(t)
            t = substitute_constants(t)

            final_result = final_result + t

        return final_result

    def build_matrix_element_expression(self, process: HardCodedProcess, expressions: dict[Any, Any], mode: EvalMode) -> Any:

        self.import_symbolica_community()
        from symbolica_community import Expression, S, E  # type: ignore # nopep8

        match mode:
            case EvalMode.TREExTREE:
                me_expr = self.sum_square_left_right(
                    expressions["tree"]["graph_expressions"], expressions["tree"]["graph_expressions"])
                me_expr = me_expr * E(process.overall_factor)
                return me_expr

            case EvalMode.LOOPxTREE:
                raise MadLoop7Error(
                    "Loop x tree matrix element expression building not implemented yet!")
            case EvalMode.LOOPxLOOP:
                raise MadLoop7Error(
                    "Loop x loop matrix element expression building not implemented yet!")

    def export_evaluator(self, process: HardCodedProcess, matrix_element_expression: Any, mode: EvalMode) -> None:
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
                    pjoin(root_path, "templates", "run_sa.py"), 'r').read()
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

    def build_tensor_networks(self, expressions: dict[Any, Any], mode=EvalMode) -> Any:

        self.import_symbolica_community()
        import symbolica_community as sc  # type: ignore # nopep8

        logger.critical("TODO: Implement tensor network building with symbolica_community!:\n{}".format(
            pformat(expressions)))
        logger.critical("# Loop diagrams: {}".format(
            len(expressions["loop"]["graph_expressions"].keys())))
        logger.critical("# Tree diagrams: {}".format(
            len(expressions["tree"]["graph_expressions"].keys())))

        tensor_nets = None  # TODO!

        return tensor_nets

    def generate(self, process_name: str = "NOT_SPECIFIED", tree_graph_ids: list[int] | None = None, loop_graph_ids: list[int] | None = None, **_opts) -> Any:

        for dir in [MADGRAPH_OUTPUT_DIR, MADSYMBOLIC_OUTPUT_DIR, GAMMALOOP_OUTPUT_DIR, MADLOOP7_OUTPUT_DIR]:
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
                process, tree_graph_ids, loop_graph_ids)

        logger.info(f"Process generation for {process_name} completed.")

    def clean(self, process_name: str) -> None:
        for graph_class in ['', '_tree', '_loop']:
            output_name = f"{process_name}{graph_class}"
            for dir in [MADGRAPH_OUTPUT_DIR, MADSYMBOLIC_OUTPUT_DIR, GAMMALOOP_OUTPUT_DIR, MADLOOP7_OUTPUT_DIR]:
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
