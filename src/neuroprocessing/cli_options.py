from pathlib import Path

import click

data_dir_option = click.option("--data-dir", type=Path, help="Path to the data directory.")
data_file_option = click.option("--filename", type=Path, help="Path to the file location.")
num_workers_option = click.option(
    "--num-workers",
    type=int,
    default=8,
    show_default=True,
    help="Number of workers for multiprocessing.",
)
