import numpy as np
# import click

from neuroprocessing.align import StackAligner

# upsampling=2, n_octaves=8, n_scales=3, sigma_min=1.6, sigma_in=0.5, c_dog=0.013333333333333334, c_edge=10, n_bins=36, lambda_ori=1.5, c_max=0.8, lambda_descr=6, n_hist=4, n_ori=8


def main(filename):
    """Wrapper for calling StackAligner.align()"""

    aligner = StackAligner(
        filepath=filename
    )

    return aligner

if __name__ == "__main__":
    from pathlib import Path

    dir_data = Path("/Users/ryanlane/Projects/neuroimaging-pruritogens/data_experimental/2024-02-21/")

    fps = [dir_data / fp for fp in [
        "Zyla_5min_LHLstim_2son4soff_1pt25pctISO_deeper_postlidocaine_2/Zyla_5min_LHLstim_2son4soff_1pt25pctISO_deeper_postlidocaine_2_MMStack_Pos0.ome.tif",
        "Zyla_5min_RHLstim_2son4soff_1pt25pctISO_deeper_1/Zyla_5min_RHLstim_2son4soff_1pt25pctISO_deeper_1_MMStack_Pos0.ome.tif"
    ]]

    aligner = main(filename=fps[0])
