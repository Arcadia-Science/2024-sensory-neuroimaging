from pathlib import Path

import click
from neuroprocessing import cli_options
from neuroprocessing.align import StackAligner

max_translation_option = click.option(
    "--max-translation",
    "max_translation",
    type=float,
    default=20,
    show_default=True,
    help="Upper limit on frame-to-frame translation.",
)
max_rotation_option = click.option(
    "--max-rotation",
    "max_rotation",
    type=float,
    default=2,
    show_default=True,
    help="Upper limit on frame-to-frame rotation.",
)
target_num_features_option = click.option(
    "--target-num-features",
    "target_num_features",
    type=int,
    default=150,
    show_default=True,
    help="Target number of features for SIFT parameter optimization.",
)
data_file_option = click.option(
    "--filename", "filepath", type=Path, help="Path to the file location."
)
num_workers_option = click.option(
    "--num-workers",
    "num_workers",
    type=int,
    default=8,
    show_default=True,
    help="Number of workers for multiprocessing.",
)


@target_num_features_option
@max_rotation_option
@max_translation_option
@num_workers_option
@data_file_option
@click.command()
def main(filepath, num_workers, max_translation, max_rotation, target_num_features):
    """Register each frame of an image stack using linear transformations.

    This script is analogous to but seeks to improve upon the Fiji plugin
    `Linear Stack Alignment with SIFT`.
    """

    aligner = StackAligner(
        filepath=filepath,
        num_workers=num_workers,
        max_translation=max_translation,
        max_rotation=max_rotation,
        target_num_features=target_num_features,
    )
    aligner.align()
    aligner.export()


if __name__ == "__main__":
    main()
