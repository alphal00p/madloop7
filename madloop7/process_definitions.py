class HardCodedProcess(object):

    def __init__(self,
                 name: str,
                 model: str,
                 madgraph_generation: str | None,
                 gamma_loop_generation: str | None,
                 madsymbolic_output: list[tuple[str, str]] | None) -> None:
        self.name = name
        self.model = model
        self.madgraph_generation = madgraph_generation
        self.gamma_loop_generation = gamma_loop_generation
        self.madsymbolic_output = madsymbolic_output


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
