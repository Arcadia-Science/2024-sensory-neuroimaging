import numpy as np
import pandas as pd


def parse_csv(
    filepath,
    columns,
    skiprows=4,
    ):
    """Parse csv file.

    csv file output by experiment looks something like
    >>>
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
    <<<
    4 columns where
    columns 0 and 2 are time (s (I think))
    column  1 is the signal from the camera (V)
    column  3 is the signal from pinch detector (V)
    """

    # read csv
    df = pd.read_csv(
        filepath,
        names=columns,
        skiprows=skiprows
    )

    # check that time columns are synchronized
    col_t1 = df.columns[0]
    col_t2 = df.columns[2]
    if (df[col_t1] != df[col_t2]).any():
        raise ValueError("Time is not synchronized between devices.")
    
    # return DataFrame sans duplicate time column
    return df.drop("time_chk", axis=1)

def process_data(df):
    """Process DataFrame.

    Returns a processed DataFrame with columns for frame number and pinch status
        frame # | pinch
        ------- | -----
              0 | False
              1 | False
            ... | ...
             37 | True
             38 | True
    """

    # estimate threshold for camera on/off signal based on voltage distribution
    # | idea is that there will be a low and high value (which are unknown)
    # | by taking a histogram we therefore expect a rather dense bin of
    # | low values, an almost empty bin of mid-range values, and a rather
    # | dense bin of high values. threshold is set at the edge of the lowest
    # | bin as long as middle bin is nearly empty (contains < 0.01% of values)
    hist, bin_edges = np.histogram(df["camera_raw"], bins=3, density=True)
    if hist[1] < 1e-4:
        df["camera"] = df["camera_raw"] > bin_edges[1]
    else:
        raise ValueError("Unable to determine threshold for camera on/off signal.")

    # do the same for pinch on/off signal
    hist, bin_edges = np.histogram(df["pinch_raw"], bins=3, density=True)
    if hist[1] < 1e-4:
        df["pinch"] = df["pinch_raw"] > bin_edges[1]
    else:
        raise ValueError("Unable to determine threshold for pinch on/off signal.")

    # get frame count from cumulative sum of positive changes to camera signal
    # basically frame count is incremented each time the camera switches back on
    # not the cleanest chain... but seems to work perty good
    df["frame"] = df["camera"]\
        .astype(int)\
        .diff()\
        .fillna(0)\
        .clip(0, None)\
        .cumsum()\
        .astype(int)

    # return processed DataFrame
    return df.loc[:, ["frame", "pinch"]].drop_duplicates()


if __name__ == "__main__":

    # packages in main only
    from pathlib import Path
    from natsort import natsorted
    from tqdm import tqdm

    # filepaths
    dir_ = Path("/Users/ryanlane/Projects/brain_imaging/data_experimental/")
    expts = "2024-02-07/A2"
    fps_csv = natsorted((dir_ / expts).glob("*/*[!flir_final].csv"))

    # parameters
    columns = [
        "time",
        "camera_raw",
        "time_chk",
        "pinch_raw"
    ]
    skiprows = 4

    # process the csvs
    for fp in tqdm(fps_csv):
    
        # parse csv
        df = parse_csv(
            filepath=fp,
            columns=columns,
            skiprows=skiprows
        )

        # process csv
        df_out = process_data(df)

        # export processed DataFrame
        fp_out = fp.parent / (fp.stem + "_processed.txt")
        df_out.to_csv(fp_out, index=False)
