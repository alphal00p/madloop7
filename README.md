# MadLoop7 path-finder

This repository contains a minimal and implementation of a different approach to computing one-loop matrix element that leverages spenso and symbolica.

The resulting low-level output for the computation of the numerator coefficient is well-suited for exploring vectorization options.

## Installation

This implenenation requires the following dependencies: `MG5aMC` equipped with the `madsymbolic` plugin, `spenso` (from `symbolica-community`) and `gammaloop`.

You can install all these dependencies by running the following commands:

```bash
cd deps && ./install_dependencies.sh
```

Alternatively, you can specify custom path to your environment-wide installation of these dependencies in `./config.yaml`.

This code has been tested with `python3.12`.

## Usage

Run `/bin/ml7.py --help` for details on the options to steer `MadLoop7`.

Also, for benchmark purposes, and access to certain utilities like a flat phase-space generator, I exported the $ g g \rightarrow g d \bar{d} $ `Python` matrix element, which you can evaluate with:

```bash
cd ./madloop7/madgraph_matrix_element/gg_gddx
python3 ./check_sa.py
```

## Example

Generate a low-level evaluator for the $ g g \rightarrow g g $ process with:

```bash
./bin/ml7.py -d generate -pn gg_gg_madgraph -lids 0 -tids 0
```

Process outputs can be cleaned up at any time with:

```bash
./bin/ml7.py clean -pn gg_gg_madgraph
```

## Adding new processes

Additional processes can easily be added to `./madloop7/process_definitions.py`, e.g:

```python
HARDCODED_PROCESSES = {
    'gg_gg_madgraph': HardCodedProcess(
        name='gg_gg_madgraph',
        model="sm-no_widths",
        madgraph_generation="""
            force_loop_model
            generate g g > g g / u c s b t [virt=QCD]
        """,
        gamma_loop_generation=None,
        madsymbolic_output=[
            ("tree", "tree_amplitude_0_gg_gg_no_ucsbt.yaml"),
            ("loop", "loop_amplitude_0_gg_gg_no_ucsbt.yaml")
        ],
    ),
}
```
