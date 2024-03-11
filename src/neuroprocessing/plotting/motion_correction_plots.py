import matplotlib.pyplot as plt
import numpy as np
from mpl_interactions import hyperslicer


def plot_motion_correction(stack_unaligned, stack_aligned):
    """Visualization for before and after comparison of motion correction.

    Layout
    ------
    top left
        Unaligned stack.
    top right
        Aligned stack.
    bottom left
        Stack of difference images for unaligned stack.
        >>> [frame[i+1] - frame[i] for i in range(len(stack_unaligned))]
    bottom right
        Stack of difference images for aligned stack.
        >>> [frame[i+1] - frame[i] for i in range(len(stack_aligned))]
    """

    # compute difference image stacks
    # recast to float for simplicity and to prevent integer overflow
    stack_unaligned_diff = stack_unaligned[1:]/65535 - stack_unaligned[:-1]/65535
    stack_aligned_diff = stack_aligned[1:]/65535 - stack_aligned[:-1]/65535
    # prepend a blank image to front of difference image stack so
    # that they have the same shape as the stacks themselves and
    # convert back to uint16
    shape = (1, *stack_unaligned.shape[1:])
    stack_unaligned_diff = np.concatenate(
        [np.zeros(shape), stack_unaligned_diff], axis=0
    )
    stack_aligned_diff = np.concatenate(
        [np.zeros(shape), stack_aligned_diff], axis=0
    )

    # create figure
    _fig, axes = plt.subplots(
        ncols=2,
        nrows=2,
        figsize=(8, 7)
    )

    # top left
    slider = hyperslicer(
        stack_unaligned,
        vmin=0,
        cmap="Greys_r",
        names="time",
        display_controls=False,
        ax=axes[0, 0]
    )

    # top right
    hyperslicer(
        stack_aligned,
        vmin=0,
        cmap="Greys_r",
        controls=slider,
        names="time",
        display_controls=False,
        ax=axes[0, 1]
    )

    # bottom left
    vmin = stack_unaligned_diff.min()
    vmax = stack_unaligned_diff.max()
    hyperslicer(
        stack_unaligned_diff,
        vmin=vmin,
        vmax=vmax,
        cmap="Greys_r",
        controls=slider,
        names="time",
        display_controls=False,
        ax=axes[1, 0]
    )

    # bottom right (use same vmin, vmax as for unaligned)
    hyperslicer(
        stack_aligned_diff,
        vmin=vmin,
        vmax=vmax,
        cmap="Greys_r",
        controls=slider,
        names="time",
        display_controls=False,
        ax=axes[1, 1]
    )

    return slider
