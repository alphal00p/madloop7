from typing import Any

import yaml  # type: ignore
import json
import os
import sys
import logging
import subprocess
import shutil
from madloop7.utils import MadLoop7Error, get_model, FlatPhaseSpaceGenerator
from madloop7.process_definitions import HardCodedProcess, HARDCODED_PROCESSES
from pprint import pformat

from . import logger

pjoin = os.path.join

MADSYMBOLIC_OUTPUT_DIR = "madsymbolic_outputs"
MADGRAPH_OUTPUT_DIR = "madgraph_outputs"
GAMMALOOP_OUTPUT_DIR = "gammaloop_outputs"
MADLOOP7_OUTPUT_DIR = "madloop7_outputs"


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
                    f"output {madgraph_output}",
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

        for graph_class in ['tree', 'loop']:
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
        for graph_class in ["tree", "loop"]:
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
                        }

            expressions[graph_class] = {
                'model_replacements': model_replacements,
                'graph_expressions': graph_expressions,
            }

        logger.debug("Model replacements:\n{}".format(
            pformat(model_replacements)))
        tensor_nets = self.build_tensor_networks(expressions)

    def build_tensor_networks(self, expressions: dict[Any, Any]) -> Any:

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
        if any(not os.path.isfile(pjoin(self.config["output_path"], GAMMALOOP_OUTPUT_DIR, f"{process.name}_{graph_class}.dot")) for graph_class in ['tree', 'loop']):
            self.generate_process(process)

        # Now process the graph with gammaLoop and spenso to get the symbolic expression
        if any(not os.path.isdir(pjoin(self.config["output_path"], GAMMALOOP_OUTPUT_DIR, f"{process.name}_{graph_class}")) for graph_class in ['tree', 'loop']):
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
