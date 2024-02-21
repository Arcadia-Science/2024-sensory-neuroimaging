import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle


def fancy_plot(
    stack,
    frames_base,
    frames_stim,
    images,
    roi,
):
    """Visualization for intrinsic imaging difference image and time signatures.

    Layout
    ------
    Top left : average of all frames during baseline (not stimulated).
    Top center : average of all frames while stimulated.
    Top right : diffrence image = `avg stim` - `avg base`.
    Bottom left & center : average intensity in ROI for each frame 
        with stimulated frames shaded orangeish.
    Bottom right : intensity distribution for baseline vs stimulated.
    """

    # create figure with `matplotlib.GridSpec`
    ncols = len(images)
    fig = plt.figure(
        constrained_layout=True,
        figsize=(4*ncols, 8)
    )
    gs = fig.add_gridspec(
        nrows=2,
        ncols=ncols,
        height_ratios=(2, 1)
    )

    # loop through images
    for i, (title, image) in enumerate(images.items()):

        # plot image
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(image, cmap="Greys_r")

        # add ROI patch
        patch = Rectangle(
            xy=(roi["center"][0] - roi["width"]//2,
                roi["center"][1] - roi["height"]//2),
            width=roi["width"],
            height=roi["height"],
            facecolor="none",
            edgecolor="#ffaa77",
            linestyle="--",
            linewidth=1.5
        )
        ax.add_patch(patch)

        # add colorbar
        div = make_axes_locatable(ax)
        cax = div.append_axes("bottom", size="5%", pad=0.35)
        fig.colorbar(im, cax=cax, orientation="horizontal")
        # aesthetics
        ax.set_title(title)

    # calculate avg intensities within ROI
    x1 = roi["center"][0] - roi["width"]//2
    x2 = roi["center"][0] + roi["width"]//2
    y1 = roi["center"][1] - roi["height"]//2
    y2 = roi["center"][1] + roi["height"]//2
    signal_roi = stack[:, y1:y2, x1:x2].mean(axis=(1, 2))
    signal_roi_base = stack[frames_base-1, y1:y2, x1:x2].mean(axis=(1, 2))
    signal_roi_stim = stack[frames_stim-1, y1:y2, x1:x2].mean(axis=(1, 2))

    # subplot for time signature within ROI (bottom left and center)
    axl = fig.add_subplot(gs[1, :2])
    # plot stimulation frames
    axl.axvline(frames_stim[0], color="#ff3300", alpha=0.05, label="stim")
    [axl.axvline(f, color="#ff3300", alpha=0.05) for f in frames_stim[1:]]
    # plot time signature within the ROI
    axl.plot(signal_roi, "k", lw=0.5)
    # aesthetics
    axl.set_title("Time signature within ROI")
    axl.grid(ls=":")
    axl.legend(loc=1)
    axl.spines["right"].set_visible(False)
    axl.spines["top"].set_visible(False)

    # subplot for before / during / after stimulation
    # TODO: ideally this would be a time series where it would show the 
    #       average intensity for the ~10 frames or so before, during,
    #       and after stimulation. but that is hard because the stimulation
    #       itself varies in number of frames. I guess you could take the
    #       shortest stimulation event and sort of trim to that, but that
    #       could be tricky and require a fair bit of bookkeeping...
    #       so for now just make it a box and whisker plot
    axd = fig.add_subplot(gs[1, 2])
    labels = ("Base", "Stim")
    axd.boxplot(
        [signal_roi_base, signal_roi_stim],
        labels=labels,
        showmeans=True,
        notch=True
    )
    # aesthetics
    axd.set_title("Base vs Stim Intensity Distribution")
    axd.grid(ls=":")
    axd.spines["right"].set_visible(False)
    axd.spines["top"].set_visible(False)
