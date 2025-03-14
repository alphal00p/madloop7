#!/usr/bin/env python3

import os  # nopep8
import sys  # nopep8
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # nopep8
from madloop7.utils import setup_logging, logger, MadLoop7Error
from madloop7.engine import MadLoop7
import argparse

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
generate_command.add_argument('--tree_graph_ids', '-tids', type=int, nargs="+", default=None,
                              help='list of tree graph ids to consider for the matrix element evaluator')
generate_command.add_argument('--loop_graph_ids', '-lids', type=int, nargs="+", default=None,
                              help='list of loop graph ids to consider for the matrix element evaluator')

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

    setup_logging(logging.DEBUG if args.debug else logging.INFO)
    madloop7 = MadLoop7(args.config)

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
