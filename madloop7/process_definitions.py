class HardCodedProcess(object):

    def __init__(self,
                 name: str,
                 model: str,
                 n_external: int,
                 madgraph_generation: str | None,
                 gamma_loop_generation: str | None,
                 madsymbolic_output: list[tuple[str, str]] | None) -> None:
        self.name = name
        self.model = model
        self.n_external = n_external
        self.madgraph_generation = madgraph_generation
        self.gamma_loop_generation = gamma_loop_generation
        self.madsymbolic_output = madsymbolic_output

    def get_graph_categories(self) -> list[str]:
        """Return the graph classes for this process."""
        possible_categories = ['tree', 'loop']
        if self.madsymbolic_output is None:
            return possible_categories
        else:
            return [category for category in possible_categories if any(
                graph_cat == category for (graph_cat, _name) in self.madsymbolic_output)]


HARDCODED_PROCESSES = {
    'gg_gg_madgraph': HardCodedProcess(
        name='gg_gg_madgraph',
        model="sm-no_widths",
        madgraph_generation="""
            force_loop_model
            generate g g > g g / u c s b t [virt=QCD]
        """,
        n_external=4,
        gamma_loop_generation=None,
        madsymbolic_output=[
            ("tree", "tree_amplitude_0_gg_gg_no_ucsbt.yaml"),
            ("loop", "loop_amplitude_0_gg_gg_no_ucsbt.yaml")
        ],
    ),
    'ddx_gg_madgraph': HardCodedProcess(
        name='ddx_gg_madgraph',
        model="sm-no_widths",
        madgraph_generation="""
            generate d d~ > g g / u c s b t
        """,
        n_external=4,
        gamma_loop_generation=None,
        madsymbolic_output=[
            ("tree", "tree_amplitude_1_ddx_gg_no_ucsbt.yaml"),
        ],
    ),
    'ddx_ssx_madgraph': HardCodedProcess(
        name='ddx_ssx_madgraph',
        model="sm-no_widths",
        madgraph_generation="""
            generate d d~ > s s~ / u c b t
        """,
        n_external=4,
        gamma_loop_generation=None,
        madsymbolic_output=[
            ("tree", "tree_amplitude_1_ddx_ssx_no_ucbt.yaml"),
        ],
    ),
}
