import click
from neuroprocessing import cli_options
from neuroprocessing.align import StackAligner

max_translation_option = click.option(
    "--max-translation",
    type=float,
    default=20,
    show_default=True,
    help="Upper limit on frame-to-frame translation.",
)
max_rotation_option = click.option(
    "--max-rotation",
    type=float,
    default=2,
    show_default=True,
    help="Upper limit on frame-to-frame rotation.",
)


@cli_options.data_file_option
@cli_options.num_workers_option
@max_translation_option
@max_rotation_option
@click.command()
def main(filename, max_translation, max_rotation, num_workers):
    """Register each frame of an image stack using linear transformations.

    This script is analogous to but seeks to improve upon the Fiji plugin
    `Linear Stack Alignment with SIFT`.
    """

    aligner = StackAligner(
        filepath=filename,
        max_translation=max_translation,
        max_rotation=max_rotation,
        num_workers=num_workers,
    )
    aligner.align()
    aligner.export()


if __name__ == "__main__":
    main()
