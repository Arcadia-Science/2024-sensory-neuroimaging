import csv
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import skimage as ski
from tqdm import tqdm


class StackAligner:
    """Align timelapse imaging data using linear transformations derived from
    SIFT + RANSAC.

    This is essentially a Python implementation of the Fiji plugin `Linear Stack
    Alignment with SIFT` [1]. The central idea is that there is bound to be some
    lateral drift between sequential frames in timelapse microscopy data. This
    class contains functionality for correcting for this drift by aligning
    sequential frames of image data with linear transformations (at the moment
    only translation + rotation are accounted for).

    The appropriate linear transformation is computed by applying a combination
    of the algorithms SIFT [2, 3] and RANSAC [4]. This is a common approach in
    computer vision. Briefly, SIFT (scale invariant feature transform) extracts
    and describes keypoints in a reference image that are robust to changes in
    scale, rotation, and translation such that it is likely to extract and
    describe many of the same keypoints in a test image containing the same
    objects but where the perspective may have changed. RANSAC (random sample
    consensus) performs geometric matching of the keypoints returned by SIFT to
    calculate the optimal linear transformation that describes this change in
    perspective. Note that the alignment can be computationally expensive
    particularly for large timelapses.

    Parameters
    ----------
    filepath : str or `pathlib.PosixPath` (optional)
        Input filename. Must be compatible with `ski.io.imread`.
    stack : ([T, Z], Y, X) array (optional)
        Input image stack as numpy array.
    num_workers : int (optional)
        Number of workers for multiprocessing.
    max_translation : scalar [px] (optional)
        Upper limit on frame-to-frame translation to apply.
    max_rotation : scalar [deg] (optional)
        Upper limit on frame-to-frame rotation to apply.
    target_num_features : int
        Target number of features for SIFT parameter optimization.
    SIFT_parameters : dict (optional)
        Parameters for SIFT (Scale Invariant Feature Transform) [5].
    RANSAC_parameters : dict (optional)
        Parameters for RANSAC (RANdom SAmple Consensus) [6].

    Examples
    --------
    Align timelapse microscopy data stored as a multipage TIFF file.
    >>> aligner = StackAligner("/path/to/timelapse.tiff")
    >>> aligner.align(max_translation=50)
    >>> aligner.stack_aligned
    array([[[5534, 5481, 5587, ..., 7950, 7813, 7937],
            [5605, 5436, 5553, ..., 8081, 7756, 7854],
            [5711, 5630, 5497, ..., 7924, 7780, 7787],
            ...,
            [8438, 8543, 8704, ..., 7327, 7072, 6984],
            [8303, 8297, 8462, ..., 7389, 7375, 7007],
            [8293, 8176, 8601, ..., 7186, 7227, 6922]]], dtype=uint16)

    Align an image stack already loaded as a numpy array with dimensions [T, Y, X].
    >>> import skimage as ski
    >>> stack = ski.io.imread("/path/to/timelapse.tiff")
    >>> aligner = StackAligner(stack=stack)
    >>> aligner.align(max_translation=50)
    >>> aligner.stack_aligned
    array([[[5534, 5481, 5587, ..., 7950, 7813, 7937],
            [5605, 5436, 5553, ..., 8081, 7756, 7854],
            [5711, 5630, 5497, ..., 7924, 7780, 7787],
            ...,
            [8438, 8543, 8704, ..., 7327, 7072, 6984],
            [8303, 8297, 8462, ..., 7389, 7375, 7007],
            [8293, 8176, 8601, ..., 7186, 7227, 6922]]], dtype=uint16)

    References
    ----------
    [1] https://imagej.net/plugins/linear-stack-alignment-with-sift
    [2] https://doi.org/10.1023/b:visi.0000029664.99615.94
    [3] https://en.wikipedia.org/wiki/Scale-invariant_feature_transform
    [4] https://en.wikipedia.org/wiki/Random_sample_consensus
    [5] https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.SIFT
    [6] https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.ransac
    """

    def __init__(
        self,
        filepath=None,
        stack=None,
        num_workers=8,
        max_translation=20,
        max_rotation=2,
        target_num_features=150,
        SIFT_parameters=None,
        RANSAC_parameters=None,
    ):
        self.load(filepath, stack)
        self.num_workers = num_workers
        self.max_translation = max_translation
        self.max_rotation = max_rotation
        self.target_num_features = target_num_features

        # reasonable SIFT parameters for expected data
        priors_SIFT = {
            "upsampling": 2,
            "n_octaves": 8,
            "n_scales": 5,
            "sigma_min": 1.6,
            "sigma_in": 0.5,
            "c_dog": 0.013333333333333334,
            "c_edge": 10,
            "n_bins": 36,
            "lambda_ori": 1.5,
            "c_max": 0.8,
            "lambda_descr": 6,
            "n_hist": 4,
            "n_ori": 8,
        }

        # this function is passed through to `ski.measure.ransac` to ensure
        # that translation and rotation stay below their limits
        is_model_valid = partial(
            validate_model, max_translation=max_translation, max_rotation=max_rotation
        )

        # reasonable RANSAC parameters for expected data
        priors_RANSAC = {
            "model_class": ski.transform.EuclideanTransform,
            "min_samples": 7,
            "residual_threshold": 2,
            "is_data_valid": None,
            "is_model_valid": is_model_valid,
            "max_trials": 500,
            "stop_sample_num": np.inf,
            "stop_residuals_sum": 0,
            "stop_probability": 1,
            "rng": None,
            "initial_inliers": None,
        }

        # overwrite SIFT + RANSAC parameters with input parameters
        SIFT_parameters = {} if SIFT_parameters is None else SIFT_parameters
        RANSAC_parameters = {} if RANSAC_parameters is None else RANSAC_parameters
        self.SIFT_parameters = {**priors_SIFT, **SIFT_parameters}
        self.RANSAC_parameters = {**priors_RANSAC, **RANSAC_parameters}

        # only allow EuclideanTransform (translation + rotation)
        model = self.RANSAC_parameters["model_class"]
        if model != ski.transform.EuclideanTransform:
            msg = f"Only `EuclideanTransform` is supported, but {model} was given."
            raise TypeError(msg)

        # these get updated once align() is run
        self.aligned = False
        self.counts_candidates = []
        self.counts_matches = []
        self.transformations = []

    def load(self, filepath, stack):
        """Load either a tiff file or numpy array.

        Tradeoff between obscurity and flexibility. Allowing a numpy array
        as input directly also has advantages for writing tests.
        """
        if filepath is not None:
            self.load_tiff(filepath)
        elif stack is not None:
            self.load_stack(stack)
        else:
            msg = (
                "StackAligner missing 1 required positional argument: either "
                "`filepath` or `stack` must be provided."
            )
            raise TypeError(msg)

    def load_tiff(self, filepath):
        """Load tiff file as numpy array.

        Tiff files are expected but will accept any image file compatible with
        `ski.io.imread`.
        """
        self.filepath = Path(filepath)
        self.stack = ski.io.imread(self.filepath)

    def load_stack(self, stack):
        """Directly load numpy array."""
        # check that input resembles an image stack
        if stack.ndim == 3:
            self.stack = stack
        else:
            msg = (
                "Expected numpy array with shape ([T, Z], Y, X) for `stack`, "
                f"but received array with shape {stack.shape}."
            )
            raise TypeError(msg)

    def optimize_SIFT_params(self, max_iterations=10):
        """Optimize SIFT parameters for feature matching.

        Tune SIFT parameters `n_scales` and `c_dog` to arrive at the target
        number of features returned by SIFT. The choice of these two parameters
        was based on ~30min of testing and is therefore somewhat arbitrary,
        hard to justify, and is subject to change in the future.

        Probably the most troubling aspect of SIFT is tuning the parameters
        to get a sensible number of features. This sensible number is
        application-dependent but normally boils down to a tradeoff between
        time and accuracy.

        For the specific application of aligning timelapse data of mouse brain
        images, 150 seems to be a sensible number from empirical observations
        thus far. Numbers higher than this will take longer without improving
        the alignment accuracy all that much. Numbers lower than this seem to
        result in less than desirable motion correction.

        NOTE: This function is very experimental
        """

        # extract the last two frames of the timelapse as they are
        # 1) guaranteed to exist and
        # 2) the most photobleached so hardest to find features from
        frame_i = self.stack[-2, :, :].copy()
        frame_j = self.stack[-1, :, :].copy()
        frame_i_preprocessed = preprocess_image_for_matching(frame_i)
        frame_j_preprocessed = preprocess_image_for_matching(frame_j)

        # get the number of features with default SIFT parameters to know
        # which direction to tune the optimization
        coords_i, _coords_j = compute_pairwise_feature_matches(
            frame_i_preprocessed, frame_j_preprocessed, self.SIFT_parameters
        )

        # begin optimization routine
        iteration_n = 0

        # too many features
        if len(coords_i) > self.target_num_features:
            while (len(coords_i) > self.target_num_features) and (iteration_n < max_iterations):
                # decrement `n_scales` every 3rd iteration and increase `c_dog`
                # by 10% every iteration to find less features
                if not iteration_n % 3:
                    self.SIFT_parameters["n_scales"] = max(1, self.SIFT_parameters["n_scales"] - 1)
                self.SIFT_parameters["c_dog"] = min(0.1, 1.1 * self.SIFT_parameters["c_dog"])

                # recompute features with adjusted SIFT parameters
                coords_i, _coords_j = compute_pairwise_feature_matches(
                    frame_i_preprocessed, frame_j_preprocessed, self.SIFT_parameters
                )

                iteration_n += 1

        # too few features
        else:
            while (len(coords_i) < self.target_num_features) and (iteration_n < max_iterations):
                # increment `n_scales` every 3rd iteration and decrease `c_dog`
                # by 10% every iteration to find more features
                if not iteration_n % 3:
                    self.SIFT_parameters["n_scales"] = min(9, self.SIFT_parameters["n_scales"] + 1)
                self.SIFT_parameters["c_dog"] = max(0.005, 0.9 * self.SIFT_parameters["c_dog"])

                # recompute features with adjusted SIFT parameters
                coords_i, _coords_j = compute_pairwise_feature_matches(
                    frame_i_preprocessed, frame_j_preprocessed, self.SIFT_parameters
                )

                iteration_n += 1

    def compute_feature_set(self):
        """Compute features for each sequential pair of images in stack.

        For each pair of images (frames) in the image stack:
            1) Preprocess them to enhance contrast for feature detection.
            2) Detect and extract SIFT features from each frame and do a
               brute-force comparison to return the subset of features
               that likely correspond with one another.
            3) Collect the x, y coordinates of potentially corresponding features.
        """

        # iterate through stack pairwise
        feature_set = {}
        for i, (frame_i, frame_j) in tqdm(
            enumerate(zip(self.stack[:-1], self.stack[1:], strict=False)), total=len(self.stack) - 1
        ):
            # get index of sequential frame
            j = i + 1

            # preprocess each frame
            frame_i_preprocessed = preprocess_image_for_matching(frame_i)
            frame_j_preprocessed = preprocess_image_for_matching(frame_j)

            # compute features
            coords_i, coords_j = compute_pairwise_feature_matches(
                frame_i_preprocessed, frame_j_preprocessed, self.SIFT_parameters
            )
            feature_set[(i, j)] = coords_i, coords_j

        return feature_set

    def compute_feature_set_parallelized(self):
        """Same as `compute_feature_set` but with multiprocessing enabled."""

        # preprocess and collect all sequential image pairs (in parallel)
        with Pool(self.num_workers) as pool:
            frames_i_preprocessed = pool.map(preprocess_image_for_matching, self.stack[:-1])
            frames_j_preprocessed = pool.map(preprocess_image_for_matching, self.stack[1:])

        # essentially duplicate parameter set for parallelized processing
        SIFT_params_batch = [self.SIFT_parameters for _ in range(len(frames_i_preprocessed))]

        # compute pairwise features (in parallel)
        with Pool(self.num_workers) as pool:
            coords_ij = pool.starmap(
                compute_pairwise_feature_matches,
                tqdm(
                    zip(
                        frames_i_preprocessed,
                        frames_j_preprocessed,
                        SIFT_params_batch,
                        strict=False,
                    ),
                    total=len(frames_i_preprocessed),
                ),
            )

        # map matched features to each pair of sequential frames
        feature_set = {(i, i + 1): v for i, v in enumerate(coords_ij)}
        return feature_set

    def align(self):
        """Align stack.

        Loop through pairs of detected features to compute the optimal
        transformation for aligning image pairs as determined by RANSAC. Then
        apply the transformation to each frame in the image stack with
        `ski.transform.warp` function.

        Notes
        -----
        * `warp` has a funky convention and applies transformations by using
          the `inverse_map` which is why `transform.inverse` is used.
        * While image intensities are rescaled for feature detection, the
          aligned stack should have same intensity range and dtype as the input
          stack.
        """
        # run optimization routine
        self.optimize_SIFT_params()

        # run feature detection either sequentially or in parallel
        if self.num_workers >= 2:
            feature_set = self.compute_feature_set_parallelized()
        else:
            feature_set = self.compute_feature_set()

        # initial transformation (null) needed to append sequential transforms
        transform = self.RANSAC_parameters["model_class"]()

        images_warped = []
        for (_i, j), (coords_i, coords_j) in feature_set.items():
            # run RANSAC
            model, inliers = ski.measure.ransac((coords_i, coords_j), **self.RANSAC_parameters)

            # TODO: properly deal with cases where RANSAC fails to find a model
            if model is None:
                msg = (
                    f"No inliers found for {len(coords_i)} potential "
                    "coordinates. Consider relaxing the constraints on maximum "
                    "allowable translation and rotation."
                )
                raise ValueError(msg)

            # keep appending transformations
            transform += model

            # apply transform to frame `j` such that it is registered to frame `i`
            frame_j = self.stack[j, :, :]
            warped = ski.transform.warp(
                image=frame_j, inverse_map=transform, preserve_range=True
            ).astype(self.stack.dtype.type)

            # collect transformation data
            self.counts_candidates.append(inliers.size)
            self.counts_matches.append(inliers.sum())
            self.transformations.append(transform)
            images_warped.append(warped)

        # create the aligned stack: the first frame + each transformed frame
        # stacked together
        self.stack_aligned = np.array([self.stack[0, :, :]] + images_warped)
        self.aligned = True

    def export(self, export_dir=None, aligned_stack=True, transformation_data=True):
        """Export aligned image stack and data regarding the transformations.

        Parameters
        ----------
        export_dir : Path (optional)
            Path to directory in which to store output. Saves output to same
            directory as input image data file by default.
        aligned_stack : bool (optional)
            Whether to export the aligned stack.
        transformation_data : bool (optional)
            Whether to export the transformation data returned by SIFT + RANSAC
            as a csv file.

        Raises
        ------
        ValueError
            If `StackAligner` was created with a numpy array AND `export_dir`
            is not provided.
        """
        # check if export directory has been provided
        if export_dir is None:
            try:
                export_dir = self.filepath.parent
            except AttributeError as err:
                msg = (
                    "`export_dir` must be provided if `StackAligner` was "
                    "created from a numpy array."
                )
                raise ValueError(msg) from err

        # check if stack has been aligned
        if not self.aligned:
            raise ValueError("Stack is not aligned.")

        if aligned_stack:
            # append "_aligned" to filename and save aligned stack as tiff file
            stem = self.filepath.stem + "_aligned"
            tgt = export_dir / (stem + ".tiff")
            ski.io.imsave(tgt, self.stack_aligned)

        if transformation_data:
            # output a csv of the form
            # TODO: put table here depicting output
            tgt = export_dir / "alignment_data.csv"
            with open(tgt, "w") as _file:
                csvwriter = csv.writer(_file)

                # write headers
                headers = [
                    "Frame ID",
                    "Translation X",
                    "Translation Y",
                    "Rotation",
                    "N candidates",
                    "N matches",
                ]
                csvwriter.writerow(headers)

                # first row is blank since first frame is not altered
                csvwriter.writerow([0 for _ in range(len(headers))])

                # iterate through lists of transform data
                id_frame = 1  # starting from 2nd frame
                for tforms, n_candidates, n_matches in zip(
                    self.transformations, self.counts_candidates, self.counts_matches, strict=False
                ):
                    # the data for each row
                    row = [
                        id_frame,
                        tforms.translation[0],
                        tforms.translation[1],
                        tforms.rotation,
                        n_candidates,
                        n_matches,
                    ]
                    csvwriter.writerow(row)
                    id_frame += 1


def validate_model(model, src, dst, max_translation, max_rotation):
    """Validation function for RANSAC with limits on translation and rotation.

    This returns a function for validating the models returned by
    RANSAC. We have observed that when running the Fiji plugin `Linear
    Stack Alignment with SIFT` (which uses the same combination of feature
    detection and transformation computing algorithms as used here), that
    there is a sort of runaway train problem in which the frame-to-frame
    transformations returned by RANSAC drive the stack further and further
    out of alignment. This is bad. Probably it is an artefact of an
    insufficient amount of features being found or that the features are
    clustered in one portion of the stack such that the transformation is
    ill-fitted. Either way, it's best to avoid that, which we attempt do
    here by defining a function that RANSAC will validate against.

    For each iteration of RANSAC it will check that the model it returns
    meets the validation criteria established here [1]. Namely, upper limits
    on the amount of allowable translation and rotation.

    References
    ----------
    [1] https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.ransac
    """
    is_valid = (
        abs(model.translation[0]) < max_translation
        and abs(model.translation[1]) < max_translation
        and abs(model.rotation) < np.deg2rad(max_rotation)
    )
    return is_valid


def preprocess_image_for_matching(image, clip_pcts=(3, 99), crop_pct=0):
    """Preprocess image for input to SIFT.

    Processing
    ----------
    1) Crop borders
    2) Enhance contrast
    3) Rescale intensity to (0, 1) range

    Parameters
    ----------
    image : (Y, X) array
        Input image
    clip_pcts : 2-tuple (optional)
        Minimum and maximum percentiles
    crop_pct : int (optional)
        Percentage (0, 100) by which to crop image borders. Has the advantages
        of speeding up feature extraction (simply because image will be smaller)
        and potentially higher fidelity features (more aberrations further away)
        from image center, but may result in less accurate registration results
        for higher amounts of motion.

    Returns
    -------
    image_rescaled : (Y, X) array
        Image with enhanced contrast and rescaled to (0, 1), for input into a feature
        detection algorithm.
    """

    # crop to image center
    ny, nx = image.shape
    b = crop_pct / 100
    dx = round(b * nx) // 2
    dy = round(b * ny) // 2
    x1, x2 = dx, nx - dx
    y1, y2 = dy, ny - dy
    image_cropped = image[y1:y2, x1:x2]

    # enhance contrast (clip intensity) and rescale to (0, 1) intensity range
    p1, p2 = np.percentile(image_cropped, clip_pcts)
    image_rescaled = ski.exposure.rescale_intensity(
        image_cropped, in_range=(p1, p2), out_range=(0, 1)
    )

    return image_rescaled


def compute_pairwise_feature_matches(image_p, image_q, SIFT_params):
    """Find matching features for an image pair using SIFT [1].

    For a pair of images (`image_p`, `image_q`), find features in each image
    using SIFT. Then do a brute-force comparison of these features to see how
    well each detected feature in `image_p` compares to each detected feature
    in `image_q`. Return the (x, y) coordinates of the subset of features
    that pass this brute-force comparison and are therefore good candidates
    for image registration.

    References
    ----------
    [1] https://doi.org/10.1109/ICCV.1999.790410
    """

    # instantiate SIFT feature detector and extractors
    features_p = ski.feature.SIFT(**SIFT_params)
    features_q = ski.feature.SIFT(**SIFT_params)
    # detect and extract SIFT features
    features_p.detect_and_extract(image_p)
    features_q.detect_and_extract(image_q)

    # brute-force matching of feature descriptors
    matches = ski.feature.match_descriptors(
        features_p.descriptors,
        features_q.descriptors,
        max_ratio=0.8,
    )

    # (x, y) pixel locations of matched features
    coords_p = features_p.keypoints[matches[:, 0]][:, ::-1]
    coords_q = features_q.keypoints[matches[:, 1]][:, ::-1]

    return coords_p, coords_q
