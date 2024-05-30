
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.exposure import rescale_intensity
from skimage.filters import gaussian
from skimage.util import montage

INJECTION_TIME = 30 # seconds, time of injection
MONTAGE_STEP = 30 # seconds, time between montage frames

def plot_STA_img_and_dff(imaging_trial,
                         trial_ROI,
                         show_montage = True,
                         flipLR = True,
                         bleach_subtract = True,
                         tiff_out = None,
                         pdf_out = None,
                         **dff_plot_kwargs):
    """
    Generate a montage or max projection image of the stimulus-triggered average (STA) of the 
    imaging trial, along with the DF/F trace of the trial.
    """
    mask = imaging_trial.load_mask()
    _, t, dff = imaging_trial.get_sta_stack(1, 5, .5, trial_ROI)
    sta_df, _, _ = imaging_trial.get_sta_stack(1, 5, .5)

    # remove linear trend from dff (remnants of bleaching)
    # only fit first and last 3 elements
    if bleach_subtract:
        dff = dff - np.polyval(np.polyfit(np.array([t[:3], t[-3:]]).flatten(),
                                        np.array([dff[:3], dff[-3:]]).flatten(), 1),
                                        t)

    # avreage over trials
    sta_df = sta_df.mean(axis=0)

    # add xy blur to sta_df image
    sta_df = gaussian(sta_df, sigma=2, channel_axis=0)

    # rescale to (0, 1), ignoring masked out areas
    sta_df = rescale_intensity(sta_df,
                               in_range=(sta_df[sta_df>0].min(), sta_df.max()),
                               out_range=(0, 1),
                               )

    # set everything outside the mask to 0
    sta_df = sta_df * mask

    if flipLR:
        sta_df = np.flip(sta_df, axis=2)

    if show_montage:
        _, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,5))
        img = axs[0].imshow(montage(sta_df,
                        fill = 0,
                        padding_width = 20,
                        rescale_intensity=True,
                        grid_shape= None,
                        ),
                    cmap='inferno',
                    aspect='equal',
                    )

    else:
        # max projection over time
        sta_df_max = sta_df.max(axis=0)

        _, axs = plt.subplots(1, 3, figsize=(18, 6), width_ratios = [.45, .02, .35]) # 6.5
        img = axs[0].imshow(sta_df_max, cmap='inferno', aspect='equal')

    if tiff_out is not None:
        io.imsave(tiff_out, (sta_df*255).astype(np.uint8))

    axs[0].axis('off')
    axs[1].axis('off')

    # plot dF/F trace
    axs[2].plot(t, dff, color='#8F8885')

    # shade the stimulus period
    axs[2].axvspan(xmin=1,
                xmax=3,
                color='#DAD3C7',
                alpha=.75,
                )
    axs[2].set_xlabel('Time (s)', fontname=plt.rcParams["font.sans-serif"][0], fontweight='medium')
    axs[2].set_ylabel('dF/F', fontname=plt.rcParams["font.sans-serif"][0], fontweight='medium')
    axs[2].tick_params(labelfontfamily=plt.rcParams["font.monospace"][0])
    axs[2].set(**dff_plot_kwargs)


    if pdf_out is not None:
        plt.savefig(pdf_out, transparent=True, bbox_inches='tight')

    return img

def plot_montage_and_trace_pairs(imaging_trials,
                                 colors,
                                 montage_grid_shape = None,
                                 trace_roi = None,
                                 trace_ylim = None,
                                 trace_xlim = None,
                                 tiff_out = None,
                                 pdf_out = None,
                                 ):
    """
    Plot montages and traces for a list of ImagingTrials. Optionally save the montage stack as tiff.
    """
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(17,11),
                            gridspec_kw={'height_ratios': [3, 1],
                                         'width_ratios': [1,1],
                                         'wspace' : .1,
                                         'hspace' : .05,
                                         }
                            )
    montage_stacks = []
    for trial,ax, color in zip(imaging_trials, axs.T, colors, strict=True):
        # montage plot
        montage_stack = trial.plot_montage(s_start = 0,
                           s_end = 1200,
                           s_step = MONTAGE_STEP,
                           montage_grid_shape = montage_grid_shape,
                           ax=ax[0],
                           cmap='inferno',
                           aspect='auto')
        ax[0].axis('off')

        # df/f plot
        t, f = trial.load_trace(roi = trace_roi)
        t0 = np.where(t >= 0)[0][0]
        f0 = f[t0-10:t0].mean()
        dff = (f - f0) / f0

        dff = dff - np.polyval(np.polyfit(t, dff, 1), t)

        ax[1].plot(t, dff, color=color['line'])

        # shade approximate injection time
        ax[1].axvspan(xmin=-INJECTION_TIME,
                      xmax=0,
                      color=color['fill'],
                      alpha=0.75,
                      )

        ax[1].set_xlim(trace_xlim)
        ax[1].set_ylim(trace_ylim)

        montage_stacks.append(montage_stack)
    axs[1,1].set_yticklabels([])
    axs[1,0].set_xlabel('Time (s)', fontname=plt.rcParams["font.sans-serif"][0], fontweight='medium')
    axs[1,1].set_xlabel('Time (s)', fontname=plt.rcParams["font.sans-serif"][0], fontweight='medium')
    axs[1,0].set_ylabel('dF/F', fontname=plt.rcParams["font.sans-serif"][0], fontweight='medium')
    axs[1,0].tick_params(labelfontfamily=plt.rcParams["font.monospace"][0])
    axs[1,1].tick_params(labelfontfamily=plt.rcParams["font.monospace"][0])

    # add colorbar
    cbar = fig.colorbar(mappable = axs[0,0].get_images()[0],
                        ax=axs,
                        location = 'right',
                        orientation = 'vertical',
                        shrink=0.5,
                        fraction = 0.05,
                        anchor = (0,1),
                        panchor = (0,1),
                        pad=0.01,
                        label='dF/F (norm.)',
                        )

    if tiff_out is not None:
        # rescale to max and min of montages in montage_stacks
        montage_stacks_np = np.vstack(montage_stacks)
        min_val, max_val = montage_stacks_np.min(), montage_stacks_np.max()
        montage_stacks = [rescale_intensity(m,
                                          in_range=(min_val, max_val),
                                          out_range=(0, 2**8-1),
                                          ) for m in montage_stacks]
        for trial, montage in zip(imaging_trials, montage_stacks, strict=True):
            io.imsave(tiff_out + str(trial) + '.tif', montage.astype(np.uint8))

    if pdf_out is not None:
        plt.savefig(pdf_out, transparent=True)
