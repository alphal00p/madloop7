class HardCodedProcess(object):

    def __init__(self,
                 name: str,
                 model: str,
                 n_external: int,
                 madgraph_generation: str | None,
                 gamma_loop_generation: list[str] | None,
                 graphs_output: list[tuple[str, str]] | None,
                 overall_factor: str | None = None) -> None:
        self.name = name
        self.model = model
        self.n_external = n_external
        self.madgraph_generation = madgraph_generation
        self.gamma_loop_generation = gamma_loop_generation
        self.graphs_output = graphs_output
        self.overall_factor = overall_factor

    def get_graph_categories(self) -> list[str]:
        """Return the graph classes for this process."""
        possible_categories = ['tree', 'loop']
        if self.graphs_output is None:
            return possible_categories
        else:
            return [category for category in possible_categories if any(
                graph_cat == category for (graph_cat, _name) in self.graphs_output)]


HARDCODED_PROCESSES_MADGRAPH = {
    'gg_gg': HardCodedProcess(
        name='gg_gg_madgraph',
        model="sm-no_widths",
        madgraph_generation="""
            force_loop_model
            generate g g > g g / u c s b t [virt=QCD]
        """,
        n_external=4,
        gamma_loop_generation=None,
        graphs_output=[
            ("tree", "tree_amplitude_0_gg_gg_no_ucsbt.yaml"),
            ("loop", "loop_amplitude_0_gg_gg_no_ucsbt.yaml")
        ],
        overall_factor="1/(8*8*2*2)",
    ),
    'gg_gg_loop': HardCodedProcess(
        name='gg_gg_loop_madgraph',
        model="sm-no_widths",
        madgraph_generation="""
            generate g g > g g / u c s b t
        """,
        n_external=4,
        gamma_loop_generation=None,
        graphs_output=[
            ("tree", "tree_amplitude_1_gg_gg_no_ucsbt.yaml"),
        ],
        overall_factor="1/(8*8*2*2)",
    ),
    'ddx_gg': HardCodedProcess(
        name='ddx_gg_madgraph',
        model="sm-no_widths",
        madgraph_generation="""
            generate d d~ > g g / u c s b t
        """,
        n_external=4,
        gamma_loop_generation=None,
        graphs_output=[
            ("tree", "tree_amplitude_1_ddx_gg_no_ucsbt.yaml"),
        ],
        overall_factor="1/(3*3*2*2)",
    ),
    'ddx_ssx': HardCodedProcess(
        name='ddx_ssx_madgraph',
        model="sm-no_widths",
        madgraph_generation="""
            generate d d~ > s s~ / u c b t
        """,
        n_external=4,
        gamma_loop_generation=None,
        graphs_output=[
            ("tree", "tree_amplitude_1_ddx_ssx_no_ucbt.yaml"),
        ],
        overall_factor="1/(3*3*2*2)",
    ),
    'epem_mupmum': HardCodedProcess(
        name='epem_mupmum_madgraph',
        model="sm-no_widths",
        madgraph_generation="""
            generate e+ e- > mu+ mu- / z
        """,
        n_external=4,
        gamma_loop_generation=None,
        graphs_output=[
            ("tree", "tree_amplitude_1_epem_mupmum_no_z.yaml"),
        ],
        overall_factor="1/(1*1*2*2)",
    ),
    'epem_epem': HardCodedProcess(
        name='epem_epem_madgraph',
        model="sm-no_widths",
        madgraph_generation="""
            generate e+ e- > e+ e- / z
        """,
        n_external=4,
        gamma_loop_generation=None,
        graphs_output=[
            ("tree", "tree_amplitude_1_epem_epem_no_z.yaml"),
        ],
        overall_factor="1/(1*1*2*2)",
    ),
    'epmup_epmup': HardCodedProcess(
        name='epmup_epmup_madgraph',
        model="sm-no_widths",
        madgraph_generation="""
            generate e+ mu+ > e+ mu+ / z
        """,
        n_external=4,
        gamma_loop_generation=None,
        graphs_output=[
            ("tree", "tree_amplitude_1_epmup_epmup_no_z.yaml"),
        ],
        overall_factor="1/(1*1*2*2)",
    ),
    'gg_ddx': HardCodedProcess(
        name='gg_ddx_madgraph',
        model="sm-no_widths",
        madgraph_generation="""
            generate g g > d d~ / u c s b t
        """,
        n_external=4,
        gamma_loop_generation=None,
        graphs_output=[
            ("tree", "tree_amplitude_1_gg_ddx_no_ucsbt.yaml"),
        ],
        overall_factor="1/(8*8*2*2)",
    ),
    'gg_ddxg': HardCodedProcess(
        name='gg_ddxg_madgraph',
        model="sm-no_widths",
        madgraph_generation="""
            generate g g > d d~ g / u c s b t
        """,
        n_external=5,
        gamma_loop_generation=None,
        graphs_output=[
            ("tree", "tree_amplitude_1_gg_ddxg_no_ucsbt.yaml"),
        ],
        overall_factor="1/(8*8*2*2)",
    ),
    'gg_ddxgg': HardCodedProcess(
        name='gg_ddxgg_madgraph',
        model="sm-no_widths",
        madgraph_generation="""
            generate g g > d d~ g g / u c s b t
        """,
        n_external=5,
        gamma_loop_generation=None,
        graphs_output=[
            ("tree", "tree_amplitude_1_gg_ddxgg_no_ucsbt.yaml"),
        ],
        overall_factor="1/(8*8*2*2)",
    ),
}

HARDCODED_PROCESSES_GAMMALOOP = {
    'gg_gg': HardCodedProcess(
        name='gg_gg_gammaloop',
        model="sm",
        madgraph_generation=None,
        n_external=4,
        gamma_loop_generation=[
            """
            generate g g > g g | g d QCD=2 [{0} QCD=0] -a -num_grouping group_identical_graphs_up_to_sign
        """,
        ],
        graphs_output=[
            ("tree", "<automatically_generated>"),
        ],
        overall_factor="1/(8*8*2*2)",
    ),
}

HARDCODED_PROCESSES = dict([
    (proc+'_madgraph', details) for proc, details in HARDCODED_PROCESSES_MADGRAPH.items()
] + [
    (proc+'_gammaloop', details) for proc, details in HARDCODED_PROCESSES_GAMMALOOP.items()
])
