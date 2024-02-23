from multiprocessing import Pool
import numpy as np
import skimage as ski

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


class StackAligner:

    def __init__(
        self,
        filepath,
        n_workers=8,
    ):
        """"""
        self.filepath = filepath
        self.n_workers = n_workers

    def load(self):
        """Load ome.tif timelapse as numpy array."""
        stack = ski.io.imread(self.filepath)
        return stack

    def compute_feature_set(self):
        """Compute image features for each sequential pair of images in stack"""

        stack = self.load()
        feature_set = {}

        # iterate through stack pairwise
        for i, (frame_1, frame_2) in tqdm(
            enumerate(zip(stack[:-1], stack[1:])),
            total=len(stack) - 1
        ):
            # get index of next frame
            j = i + 1

            # preprocess each frame
            f1 = preprocess_image_for_matching(frame_1)
            f2 = preprocess_image_for_matching(frame_2)

            # compute features
            kps1, kps2 = compute_pairwise_feature_matches(f1, f2)
            feature_set[(i, j)] = kps1, kps2

        return feature_set
    
    def compute_feature_set_parallelized(self):
        """Compute image features for each sequential pair of images in stack"""

        # load stack
        stack = self.load()

        # preprocess and collect all sequential image pairs (in parallel)
        with Pool(self.n_workers) as pool:
            f1s = pool.map(preprocess_image_for_matching, stack[:-1, :, :])
            f2s = pool.map(preprocess_image_for_matching, stack[1:, :, :])

        # compute pairwise features (in parallel)
        with Pool(self.n_workers) as pool:
            results = pool.starmap(
                compute_pairwise_feature_matches,
                tqdm(zip(f1s, f2s), total=len(f1s))
            )

        # map matched features to each pair of sequential frames
        feature_set = {(k, k+1): v for k, v in enumerate(results)}
        return feature_set

    def align(self):
        """Align stack"""
        if self.n_workers >= 2:
            feature_set = self.compute_feature_set_parallelized()
        else:
            feature_set = self.compute_feature_set()
        
        stack_aligned = feature_set
        return stack_aligned


def preprocess_image_for_matching(
    image,
    clip_pcts=(1, 99),
    crop_pct=80
):
    """Preprocess image for input to SIFT"""

    # crop to center
    ny, nx = image.shape
    dx = round((1 - crop_pct/100) * nx) // 2
    dy = round((1 - crop_pct/100) * ny) // 2
    x1, x2 = dx, nx - dx
    y1, y2 = dy, ny - dy
    image_cropped = image[y1:y2, x1:x2]

    # enhance contrast (clip intensity) and 
    # rescale to (0, 1) intensity range
    p1, p2 = np.percentile(image, clip_pcts)
    image_rescaled = ski.exposure.rescale_intensity(
        image_cropped,
        in_range=(p1, p2),
        out_range="float"
    )

    return image_rescaled


def compute_pairwise_feature_matches(
    image_a,
    image_b,
    SIFT_params=None,
):
    """Find matching features for an image pair using SIFT [1].

    References
    ----------
    [1] https://doi.org/10.1109/ICCV.1999.790410
    """
    # handle args
    SIFT_priors = {
        "upsampling": 2,
        "n_octaves": 8,
        "n_scales": 3,
        "sigma_min": 1.6
    }
    SIFT_params = {} if SIFT_params is None else SIFT_params
    SIFT_params = {**SIFT_priors, **SIFT_params}

    # instantiate SIFT feature detector and extractors
    fde_a = ski.feature.SIFT(**SIFT_params)
    fde_b = ski.feature.SIFT(**SIFT_params)
    # detect and extract SIFT features
    fde_a.detect_and_extract(image_a)
    fde_b.detect_and_extract(image_b)

    # brute-force matching of feature descriptors
    matches = ski.feature.match_descriptors(
        fde_a.descriptors,
        fde_b.descriptors,
        max_ratio=0.8
    )
    # (r, c) pixel locations of matched features
    coords_a = fde_a.keypoints[matches[:, 0]]
    coords_b = fde_b.keypoints[matches[:, 1]]

    return coords_a, coords_b


def compute_transform_from_matched_features(
    coords_a,
    coords_b,
    RANSAC_params=None
):
    """"""
    # handle args
    RANSAC_priors = {
        "model_class": ski.transform.EuclideanTransform,
        "min_samples": 7,
        "residual_threshold": 10,
        "max_trials": 1000
    }
    RANSAC_params = {} if RANSAC_params is None else RANSAC_params
    RANSAC_params = {**RANSAC_priors, **RANSAC_params}

    # use RANSAC to compute transform from feature matches
    src = coords_a
    dst = coords_b
    model, inliers = ski.measure.ransac(
        (src, dst),
        **RANSAC_params
    )

    return model


def compute_transform(
    image_a,
    image_b,
    SIFT_params=None,
    RANSAC_params=None
):
    """"""
    # handle args
    RANSAC_priors = {
        "model_class": ski.transform.EuclideanTransform,
        "min_samples": 7,
        "residual_threshold": 10,
        "max_trials": 1000
    }
    RANSAC_params = {} if RANSAC_params is None else RANSAC_params
    RANSAC_params = {**RANSAC_priors, **RANSAC_params}

    # run SIFT on image pair to compute feature matches
    src, dst = compute_pairwise_feature_matches(
        image_a,
        image_b,
        SIFT_params
    )

    # use RANSAC to compute transform from feature matches
    model, inliers = ski.measure.ransac(
        (src, dst),
        **RANSAC_params
    )

    return model
