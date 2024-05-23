import csv

import numpy as np
import pandas as pd


def parse_csv(
    filepath,
    columns=None,
    row_channels=3,
    skiprows=4,
):
    """Parse csv file output by NI DAQ.

    csv file output by experiment looks something like

        Timestamp,2/7/2024 3:58:30 PM,Timestamp,2/7/2024 3:58:30 PM
        Interval,0.0001,Interval,0.0001
        Channel name,"Frame",Channel name,"Button"
        Unit,"V",Unit,"V"
        0,0.13845624891109765,0,0.025044171372428536
        0.0001,0.13845624891109765,0.0001,0.043087001889944077
        0.0002,0.13845624891109765,0.0002,0.050819643540307879
        0.0003,0.13845624891109765,0.0003,0.049530869931913912
        0.0004,0.13845624891109765,0.0004,0.046953322715125978
        0.0005,0.13845624891109765,0.0005,0.053397190757095814
        0.0006,0.13716747530270368,0.0006,0.050819643540307879
        0.0007,0.13845624891109765,0.0007,0.049530869931913912
        0.0008,0.13845624891109765,0.0008,0.053397190757095814

    where columns alternate between time and input channel
        Time, Channel 1, Time, Channel 2, ..., Time, Channel N
    """
    # infer column names from csv header if not provided
    if columns is None:
        with open(filepath) as f:
            r = csv.reader(f)
            # csv is usually huge so only read up to row {row_channels}
            # (where the name of each channel should be)
            for _i in range(row_channels):
                channels = next(r)

        # channels now resembles
        #     'Channel name', 'camera', 'Channel name', ...
        # but pandas does not allow for duplicates so
        # have to create unique column names
        columns = []
        for i, name in enumerate(channels):
            # rather hacky but only even columns are duplicates (all
            # 'Channel name') so rename those to time-0, time-2, etc.
            column = name if i % 2 else f"time-{i}"
            columns.append(column)

    # read csv
    df = pd.read_csv(filepath, names=columns, skiprows=skiprows)

    # check that all time columns are synchronized
    for col_time in columns[2::2]:
        if (df.loc[:, columns[0]] != df.loc[:, col_time]).any():
            raise ValueError("Time is not synchronized between devices.")
        # drop (duplicate) column
        df.drop(col_time, axis=1, inplace=True)

    # rename time column and return DataFrame
    return df.rename({df.columns[0]: "time"}, axis=1)


def process_data(df, col_camera="camera", col_stim="stim"):
    """Process DataFrame.

    Returns a processed DataFrame with columns for frame number,
    stimulated status, and frame rate.
        frame | stimulated | time | frametime | framerate
        ----- | ---------- | ---- | --------- | ---------
            1 |      False | 2.88 |    0.0565 | 17.699115
            2 |      False | 2.93 |    0.0565 | 17.699115
            3 |      False | 2.99 |    0.0565 | 17.699115
            4 |      False | 3.04 |    0.0565 | 17.699115
    """

    # estimate threshold for camera on/off signal based on voltage distribution
    # | idea is that there will be a low and high value (which are unknown)
    # | by taking a histogram we therefore expect a rather dense bin of
    # | low values, an almost empty bin of mid-range values, and a rather
    # | dense bin of high values. threshold is set at the edge of the lowest
    # | bin as long as middle bin is the least populated.
    hist, bin_edges = np.histogram(df.loc[:, col_camera], bins=3, density=True)
    if hist.min() == hist[1]:
        df["camera_BOOL"] = df.loc[:, col_camera] > bin_edges[1]
    else:
        raise ValueError("Unable to determine threshold for camera on/off signal.")

    # do the same for stimulus on/off signal
    hist, bin_edges = np.histogram(df.loc[:, col_stim], bins=3, density=True)
    if hist.min() == hist[1]:
        df["stimulated"] = df.loc[:, col_stim] > bin_edges[1]
    else:
        msg = (
            "Unable to determine threshold for stimulus on/off signal. "
            f"Is the stimulus signal `{col_stim}` correct?"
        )
        raise ValueError(msg)

    # get frame count from cumulative sum of positive changes to camera signal
    # basically frame count is incremented each time the camera switches back on
    df["frame"] = df["camera_BOOL"].astype(int).diff().fillna(0).clip(0, None).cumsum().astype(int)

    # create new DataFrame where each row marks the start of a new frame
    df_frames = df.loc[df["frame"].diff().fillna(0) > 0, ["frame", "stimulated", "time"]]
    # calculate frame time and frame rate
    df_frames["frametime"] = df_frames["time"].diff().shift(-1).fillna(-1).round(7)
    df_frames["framerate"] = (1 / df_frames["frametime"]).round(7)

    return df_frames
