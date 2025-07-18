#!/usr/bin/env python3

import os  # nopep8
import sys  # nopep8
import logging
import yaml  # type: ignore # nopep8
import argparse
import multiprocessing

parser = argparse.ArgumentParser(prog='MadLoop7')
parser.add_argument('--config', '-c', type=str, default='config.yaml',
                    help='Path to the configuration file')
parser.add_argument('--debug', '-d', default=False, action='store_true',
                    help='Enable debug logging')

subparsers = parser.add_subparsers(
    title="Command to run", dest="command", required=True,
    help='Various commands for MadLoop7 to run')

# Create the parser for the "epem_lplm_fixed_order_LO" experiment
generate_command = subparsers.add_parser(
    'generate', help='Generate a process')

generate_command.add_argument('--process_name', '-pn', type=str, default='gg_gg_madgraph',
                              help='Hard-coded process name to generate')
generate_command.add_argument('--evaluation_strategy', '-es', type=str, default='tensor_networks',
                              choices=['only_dot_products', 'tensor_networks'], help='Evaluation strategy.')
generate_command.add_argument('--tree_graph_ids', '-tids', type=int, nargs="+", default=None,
                              help='list of tree graph ids to consider for the matrix element evaluator')
generate_command.add_argument('--loop_graph_ids', '-lids', type=int, nargs="+", default=None,
                              help='list of loop graph ids to consider for the matrix element evaluator')

generate_command.add_argument('--inline_asm', '-asm', type=str, default="default", choices=["default", "none"],
                              help="Whether to enable inline assembly or not during symbolica's codegen. default=%(default)s")
generate_command.add_argument('--optimisation_level', '-O', type=int, default=3,
                              help="Optimization level for symbolica's codegen. default=%(default)s")
generate_command.add_argument('--targets', '-t', type=str, nargs="+", default=[],
                              help='Targets for the evaluations of this process. Specify as list of strings, i.e. "0.12+0.45j".')
generate_command.add_argument('--n_cores', '-c', type=int, default=multiprocessing.cpu_count(),
                              help='Number of cores for generating the expression and building the evaluator. default=%(default)s')
generate_command.add_argument('--work_in_emr', '-emr', type=bool, default=True, action=argparse.BooleanOptionalAction,
                              help='Whether to work in the Edge Momentum Representation when building the matrix element expression. default=%(default)s')
generate_command.add_argument('--simplify_amplitude_gamma_algebra', '-saga', type=bool, default=False, action=argparse.BooleanOptionalAction,
                              help='Whether to simplify the gamma algebra at the amplitude level before building the expression for the squared matrix element. default=%(default)s')
generate_command.add_argument('--step_through_network_execution', '-step', type=bool, default=False, action=argparse.BooleanOptionalAction,
                              help='Whether to step through the tensor network execution. default=%(default)s')
generate_command.add_argument('--expand_before_building_evaluator', '-exp', type=bool, default=False, action=argparse.BooleanOptionalAction,
                              help='Expand expression before building evaluator in Symbolica. default=%(default)s')
generate_command.add_argument('--physical_vector_polarization_sum', '-phys', type=bool, default=True, action=argparse.BooleanOptionalAction,
                              help='Use physical vector polarisation sums for vectors, so that external ghosts are not necessary. default=%(default)s')

clean_command = subparsers.add_parser(
    'clean', help='Clean a process output')

clean_command.add_argument('--process_name', '-pn', type=str, default='gg_gg_madgraph',
                           help='Hard-coded process name to clean')

install_dependencies_command = subparsers.add_parser(
    'install_dependencies', help='Install all MadLoop7 dependencies')

install_dependencies_command.add_argument('--symbolica_community', default=False, action=argparse.BooleanOptionalAction,
                                          help='Include symbolica community dependencies.')

playground = subparsers.add_parser(
    'playground', help='Run custom debug code')

if __name__ == "__main__":

    args = parser.parse_args()

    if args.command == 'generate':
        evaluated_targets = []
        for t in args.targets:
            try:
                evaluated_targets.append(eval(t))
            except Exception as e:
                raise ValueError(
                    f"Could not evaluate target to a complex value '{t}': {e}")
        args.targets = evaluated_targets

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    os.environ['SYMBOLICA_COMMUNITY_PATH'] = config['symbolica_community_path']

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # nopep8
    from madloop7.utils import setup_logging, logger, MadLoop7Error
    from madloop7.engine import MadLoop7

    setup_logging(logging.DEBUG if args.debug else logging.INFO)

    madloop7 = MadLoop7(args.config, **args.__dict__)

    match args.command:
        case 'generate':
            madloop7.generate(**{k: getattr(args, k)
                                 for k in dir(args) if not k.startswith('_')})

        case 'clean':
            madloop7.clean(args.process_name)

        case 'install_dependencies':
            madloop7.install_dependencies(
                symbolica_community=args.symbolica_community)

        case 'playground':
            madloop7.playground()

        case _:
            raise MadLoop7Error(
                f"Command {args.command} not implemented")
