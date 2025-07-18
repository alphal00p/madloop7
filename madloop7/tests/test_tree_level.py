import pytest
import subprocess
import os
import glob
import shutil

ml7_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))  # nopep8
output_path = os.path.join(ml7_path, "outputs", "madloop7_outputs")


def run_test(pocess_name, ml7_options, clean=False):

    if clean:
        cmd = ['./bin/ml7.py', 'clean', '-pn', f'{pocess_name}']
        subprocess.run(cmd, cwd=ml7_path, check=True)  # nopep8
        assert glob.glob(os.path.join(ml7_path, "outputs", "*", f"*{pocess_name}*")) == []  # nopep8

    if os.path.isdir(os.path.join(output_path, pocess_name)):
        shutil.rmtree(os.path.join(output_path, pocess_name))

    cmd = [
        './bin/ml7.py',
        'generate',
        '-pn', pocess_name,
    ] + ml7_options

    subprocess.run(cmd, cwd=ml7_path, check=True)  # nopep8
    assert os.path.isfile(os.path.join(output_path, pocess_name, 'run.py'))  # nopep8
    subprocess.run(['python3', './run.py'], cwd=os.path.join(output_path, pocess_name), check=True)  # nopep8

    assert open(os.path.join(output_path, pocess_name, 'TEST_OUTCOME.txt'), 'r').read() == "SUCCESS"  # nopep8


def test_gg_ddx():
    """ MadGraph runs this process in 2.5 mus """

    # 0.669 mus | 53 ×
    run_test('gg_ddx_gammaloop',
             [
                 '-es', 'only_dot_products',
                 '--no-physical_vector_polarization_sum',
                 '--targets', '0.4812176671899552+0j', '0.5003352376819938+0j',
                 '-asm', 'default'
             ]
             )

    # 3.202 mus | 4181 ×
    run_test('gg_ddx_gammaloop',
             [
                 '-es', 'tensor_networks',
                 '--no-physical_vector_polarization_sum',
                 '--targets', '0.4812176671899552+0j', '0.5003352376819938+0j',
                 '-asm', 'default'
             ]
             )
    # 0.898 mus | 801 ×
    run_test('gg_ddx_gammaloop',
             [
                 '-es', 'only_dot_products',
                 '--physical_vector_polarization_sum',
                 '--targets', '0.38386585832220405+0j', '0.40468649911974430+0j',
                 '-asm', 'default'
             ],
             clean=True
             )
    # 35.066 mus | 5948 ×
    run_test('gg_ddx_gammaloop',
             [
                 '-es', 'tensor_networks',
                 '--physical_vector_polarization_sum',
                 '--targets', '0.38386585832220405+0j', '0.40468649911974430+0j',
                 '-asm', 'default'
             ]
             )


def test_ddx_ssxuux():
    """ MadGraph runs this process in 6.9 mus """

    # 114.956 mus | 80721 ×
    run_test('ddx_ssxuux_gammaloop',
             [
                 '-es', 'tensor_networks',
                 '--targets', '9.0399300845323664e-10+0j', '1.1620030447906114e-9+0j',
                 '-asm', 'default'
             ]
             )

    # 3.543 mus | 6140 ×
    run_test('ddx_ssxuux_gammaloop',
             [
                 '-es', 'only_dot_products',
                 '--targets', '9.0399300845323664e-10+0j', '1.1620030447906114e-9+0j',
                 '-asm', 'default'
             ]
             )


def test_gg_ddxg_first_two_graphs():
    """ MadGraph runs this process in """

    run_test('gg_ddxg_gammaloop',
             [
                 '-es', 'only_dot_products',
                 '--no-physical_vector_polarization_sum',
                 '--targets', '0.0005280635119873812+0j', '0.05521549665831862+0j',
                 '-asm', 'default',
                 '-tids', '0', '1', '2'  # Only the first two graphs
             ]
             )

    # Cannot run now as it hangs
    # run_test('gg_ddxg_gammaloop',
    #          [
    #              '-es', 'tensor_networks',
    #              '--no-physical_vector_polarization_sum',
    #              '--targets', '0.0005280635119873812+0j', '0.05521549665831862+0j',
    #              '-asm', 'default',
    #              '-tids', '0', '1', '2'  # Only the first two graphs
    #          ]
    #          )

    run_test('gg_ddxg_gammaloop',
             [
                 '-es', 'only_dot_products',
                 '--physical_vector_polarization_sum',
                 '--targets', '7.9531811004898744e-5+0j', '3.6839654074578530e-2+0j',
                 '-asm', 'default',
                 '-tids', '0', '1'  # Only the first two graphs
             ]
             )

    # Cannot run now as it hangs
    # run_test('gg_ddxg_gammaloop',
    #          [
    #              '-es', 'tensor_networks',
    #              '--physical_vector_polarization_sum',
    #              '--targets', '7.9531811004898744e-5+0j', '3.6839654074578530e-2+0j',
    #              '-asm', 'default',
    #              '-tids', '0', '1'  # Only the first two graphs
    #          ]
    #          )


@ pytest.mark.skip(reason="This is currently too slow for only_dot_products and tensor_networks is bugged")
def test_skip_me():
    assert 1 + 1 == 3
