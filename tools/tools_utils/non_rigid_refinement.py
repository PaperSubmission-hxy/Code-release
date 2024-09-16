"""Perform non-rigid registration
"""

import os
import pathlib
import numpy as np
import SimpleITK as sitk
from .plot import blend_img

NR_CLS_KEY = "non_rigid_registrar_cls"
NR_PROCESSING_KW_KEY = "processer_kwargs"
NR_PROCESSING_INIT_KW_KEY = "init_processer_kwargs"
NR_PROCESSING_CLASS_KEY = "processer_cls"
NR_STATS_KEY = "target_stats"
NR_TILE_WH_KEY = "tile_wh"
NR_PARAMS_KEY = "params"

NR_MOVING = "moving"
NR_TILE_MOVING_P_KEY = f"{NR_MOVING}_{NR_PROCESSING_CLASS_KEY}"
NR_TILE_MOVING_P_INIT_KW_KEY = f"{NR_MOVING}_{NR_PROCESSING_INIT_KW_KEY}"
NR_TILE_MOVING_P_KW_KEY = f"{NR_MOVING}_{NR_PROCESSING_KW_KEY}"

NR_FIXED = "fixed"
NR_TILE_FIXED_P_KEY = f"{NR_FIXED}_{NR_PROCESSING_CLASS_KEY}"
NR_TILE_FIXED_P_INIT_KW_KEY = f"{NR_FIXED}_{NR_PROCESSING_INIT_KW_KEY}"
NR_TILE_FIXED_P_KW_KEY = f"{NR_FIXED}_{NR_PROCESSING_KW_KEY}"

# Abstract Classes #
class NonRigidRegistrar(object):
    """Abstract class for non-rigid registration using displacement fields

    Warps moving_img to align with fixed_img using backwards transformations
    VALIS offers 3 implementations: dense optical flow (OpenCV),
    SimpleElastix, and  groupwise SimpleElastix.
    Displacement fields can come from other packages, indluding
    SimpleITK, PIRT, DIPY, etc... Those other methods can be used by
    subclassing the NonRigidRegistrar classes in VALIS.

    Attributes
    ----------
    moving_img : ndarray
        Image with shape (N,M) thata is  warp to align with `fixed_img`.

    fixed_img : ndarray
        Image with shape (N,M) that `moving_img` is warped to align with.

    mask : ndarray
        2D array with shape (N,M) where non-zero pixel values are foreground,
        and 0 is background, which is ignnored during registration. If None,
        then all non-zero pixels in images will be used to create the mask.

    shape : tuple
        Number of rows and columns in each image. Will be (N,M).

    grid_spacing : int
        Number of pixels between deformation grid points.

    warped_image : ndarray
        Registered copy of `moving_img`.

    deformation_field_img : ndarray
        Image showing deformation applied to a regular grid.

    backward_dx : ndarray
        (N,M) array defining the displacements in the x-dimension.

    backward_dy : ndarray
        (N,M) array defining the displacements in the y-dimension.

    method : str
        Name of registration method.

    Note
    -----
    All NonRigidRegistrar subclasses need to have a calc() method,
    which must at least take the following arguments:
    moving_img, fixed_img, mask. calc() should return the displacement field
    as a (2, N, M) numpy array, with the first element being an array of
    displacements in the x-dimension, and the second element being an array of
    displacements in the y-dimension.

    Note that the NonRigidRegistrarXY subclass should be used if
    corresponding points in moving and fixed images can be used
    to aid the registration.

    """

    def __init__(self, params=None):
        """
        Parameters
        ----------
        params : dictionary
            Keyword: value dictionary of parameters to be used in reigstration.
            Will get used in the calc() method.

            In the case where simple ITK will be used, params should be
            a SimpleITK.ParameterMap. Note that numeric values needd to be
            converted to strings.

        """

        self.params = params
        self.moving_img = None
        self.fixed_img = None
        self.mask = None
        self.shape = None
        self.grid_spacing = None
        self.method = None
        self.warped_image = None
        self.deformation_field_img = None
        self.backward_dx = None
        self.backward_dy = None

        if isinstance(params, dict):
            if len(params) > 0:
                self._params_provided = True
            else:
                # Empty kwargs dictionary
                self._params_provided = False
        else:
            self._params_provided = False

    def apply_mask(self, mask):

        masked_moving = warp_tools.apply_mask(self.moving_img, mask)
        masked_fixed = warp_tools.apply_mask(self.fixed_img, mask)

        return masked_moving, masked_fixed

    def calc(self, moving_img, fixed_img, mask, *args, **kwargs):
        """Cacluate displacement fields

        Can record subclass specific atrributes here too

        Parameters
        ----------
        moving_img : ndarray
            Image to warp to align with `fixed_img`. Has shape (N, M).

        fixed_img : ndarray
            Image `moving_img` is warped to align with. Has shape (N, M).

        mask : ndarray
            2D array with shape (N,M) where non-zero pixel values are foreground,
            and 0 is background, which is ignnored during registration. If None,
            then all non-zero pixels in images will be used to create the mask.

        Returns
        -------
        bk_dxdy : ndarray
            (2, N, M) numpy array of pixel displacements in
            the x and y directions. dx = bk_dxdy[0], and dy=bk_dxdy[1].

        """

        bk_dxdy = None

        return bk_dxdy

    def create_mask(self):
        temp_mask = np.zeros(self.shape, dtype=np.uint8)
        img_list = [self.moving_img, self.fixed_img]
        for img in img_list:
            temp_mask[img > 0] = 255

        mask = warp_tools.bbox2mask(*warp_tools.xy2bbox(
                                    warp_tools.mask2xy(temp_mask)),
                                    temp_mask.shape)

        return mask

    def register(self, moving_img, fixed_img, mask=None, **kwargs):
        """
        Register images, warping moving_img to align with fixed_img

        Uses backwards transforms to register images (i.e. aligning
        fixed to moving), so the inverse transform needs to be used
        to warp points from moving_img. This is automatically done in
        warp_tools.warp_xy

        Parameters
        ----------
        moving_img : ndarray
            Image to warp to align with `fixed_img`.

        fixed_img : ndarray
            Image `moving_img` is warped to align with.

        mask : ndarray
            2D array with shape (N,M) where non-zero pixel values are foreground,
            and 0 is background, which is ignnored during registration. If None,
            then all non-zero pixels in images will be used to create the mask.

        **kwargs : dict, optional
            Additional keyword arguments passed to NonRigidRegistrar.calc

        Returns
        -------
        warped_img : ndarray
            Moving image registered to align with fixed image.

        warped_grid : ndarray
            Image showing deformation applied to a regular grid.

        bk_dxdy : ndarray
            (2, N, M) numpy array of pixel displacements in
            the x and y directions.

        """

        moving_shape = warp_tools.get_shape(moving_img)[0:2]
        fixed_shape = warp_tools.get_shape(fixed_img)[0:2]

        assert np.all(moving_shape == fixed_shape), \
            print("Images have different shapes")

        self.shape = moving_shape
        self.moving_img = moving_img
        self.fixed_img = fixed_img

        if mask is None:
            mask = np.full(self.shape, 255, dtype=np.uint8)

        self.mask = mask

        if self.mask is not None:
            # Only do registration inside mask #
            _, masked_fixed = self.apply_mask(self.mask)
            masked_moving = self.moving_img.copy()

            mask_bbox = warp_tools.xy2bbox(warp_tools.mask2xy(self.mask))
            min_c, min_r = mask_bbox[0:2]
            max_c, max_r = mask_bbox[0:2] + mask_bbox[2:]
            mask = self.mask[min_r:max_r, min_c:max_c]
            masked_moving = masked_moving[min_r:max_r, min_c:max_c]
            masked_fixed = masked_fixed[min_r:max_r, min_c:max_c]

        else:
            masked_moving = self.moving_img.copy()
            masked_fixed = self.fixed_img.copy()

        bk_dxdy = self.calc(moving_img=masked_moving,
                            fixed_img=masked_fixed,
                            mask=mask, **kwargs)

        if mask is not None:
            bk_dx = np.zeros(self.shape)
            bk_dx[min_r:max_r, min_c:max_c] = bk_dxdy[0]
            bk_dx[self.mask == 0] = 0

            bk_dy = np.zeros(self.shape)
            bk_dy[min_r:max_r, min_c:max_c] = bk_dxdy[1]
            bk_dy[self.mask == 0] = 0

            bk_dxdy = np.array([bk_dx, bk_dy])

        warped_img, warp_grid = self.get_warped_img_and_grid(bk_dxdy)

        self.backward_dx = bk_dxdy[..., 0]
        self.backward_dy = bk_dxdy[..., 1]
        self.deformation_field_img = warp_grid
        self.warped_image = warped_img

        return warped_img, warp_grid, bk_dxdy

    def get_grid_image(self, grid_spacing=None, thickness=1, grid_spacing_ratio=0.025):
        """Create an image of a regular grid.

        Usually used to visualize non-rigid deformations.

        Parameters
        ----------
        grid_spacing : int, optional
            Number of pixels between grid points.

        thickness : int, optional
            Thickness of lines in image.

        """

        if grid_spacing is None:
            if self.grid_spacing is not None:
                grid_spacing = int(self.grid_spacing)
            else:
                grid_spacing = np.max(np.array(self.shape)*grid_spacing_ratio).astype(int)

        grid_r, grid_c = viz.get_grid(self.shape[:2],
                                      grid_spacing=grid_spacing,
                                      thickness=thickness)

        grid_img = np.zeros(self.shape[:2])
        grid_img[grid_r, grid_c] = 255

        return grid_img

    def get_warped_img_and_grid(self, bk_dxdy):
        """Apply deformation to moving image and regular grid

        Parameters
        ----------
        bk_dxdy : ndarray
            (2, N, M) numpy array of pixel displacements in
            the x and y directions.

        Returns
        -------
        warped_img : ndarray
            Warped copy of moving image.

        warp_grid : ndarray
            Image showing deformation applied to regular grid.

        """

        warp_map = warp_tools.get_warp_map(dxdy=bk_dxdy)
        warped_img =transform.warp(self.moving_img, warp_map, preserve_range=True)
        self.warped_image = warped_img
        grid_img = self.get_grid_image(grid_spacing=16)
        warp_grid = transform.warp(grid_img, warp_map, preserve_range=True)

        return warped_img, warp_grid


class NonRigidRegistrarXY(NonRigidRegistrar):
    """Abstract class for non-rigid registration using displacement fields

    Subclass of NonRigidRegistrar that can (optionally) use corresponding
    points (xy coordinates) to aid in the registration

    Attributes
    ----------
    moving_img : ndarray
        Image with shape (N,M) thata is  warp to align with `fixed_img`.

    fixed_img : ndarray
        Image with shape (N,M) that `moving_img` is warped to align with.

    mask : ndarray
        2D array with shape (N,M) where non-zero pixel values are foreground,
        and 0 is background, which is ignnored during registration. If None,
        then all non-zero pixels in images will be used to create the mask.

    shape : tuple
        Number of rows and columns in each image. Will be (N,M).

    grid_spacing : int
        Number of pixels between deformation grid points/

    warped_image : ndarray
        Registered copy of `moving_img`.

    deformation_field_img : ndarray
        Image showing deformation applied to a regular grid.

    backward_dx : ndarray
        (N,M) array defining the displacements in the x-dimension.

    backward_dy : ndarray
        (N,M) array defining the displacements in the y-dimension.

    method : str
        Name of registration method.

    moving_xy : ndarray, optional
        (N, 2) array containing points in `moving_img` that correspond
        to those in the fixed image.

    fixed_xy : ndarray, optional
        (N, 2) array containing points in `fixed_img` that correspond
        to those in the moving image.

    Note
    ----
    All NonRigidRegistrarXY subclasses need to have a calc() method,
    which needs to at least take the following arguments:
    moving_img, fixed_img, mask, moving_xy, fixed_xy.
    calc() should return the warped moving image, warped regular grid,
    and the displacement field as an (2, N, M) numpy array.

    Note that NonRigidRegistrar should be used if corresponding points in
    moving and fixed images can not be used to aid the registration.

    """

    def __init__(self, params=None):
        super().__init__(params=params)
        """
        Parameters
        ----------
        params : dictionary
            Keyword: value dictionary of parameters to be used in reigstration.
            Will get used in the calc() method.

            In the case where simple ITK will be used, params should be
            a SimpleITK.ParameterMap. Note that numeric values needd to be
            converted to strings.

        moving_xy : ndarray, optional
            (N, 2) array containing points in the moving image that correspond
            to those in the fixed image.

        fixed_xy : ndarray, optional
            (N, 2) array containing points in the fixed image that correspond
            to those in the moving image.

        """

        self.moving_xy = None
        self.fixed_xy = None

    def register(self, moving_img, fixed_img, mask=None, moving_xy=None,
                 fixed_xy=None, **kwargs):
        """Register images, warping moving_img to align with fixed_img

        Uses backwards transforms to register images (i.e. aligning
        fixed to moving), so the inverse transform needs to be used
        to warp points from moving_img. This is automatically done in
        warp_tools.warp_xy

        Parameters
        ----------
        moving_img : ndarray
            Image to warp to align with `fixed_img`.

        fixed_img : ndarray
            Image `moving_img` is warped to align with.

        mask : ndarray
            2D array with shape (N,M) where non-zero pixel values are foreground,
            and 0 is background, which is ignnored during registration. If None,
            then all non-zero pixels in images will be used to create the mask.

        moving_xy : ndarray, optional
            (N, 2) array containing points in the `moving_img` that correspond
            to those in `fixed_img`.

        fixed_xy : ndarray, optional
            (N, 2) array containing points in the `fixed_img` that correspond
            to those in the `moving_img`.

        Returns
        -------
        warped_img : ndarray
            `moving_img` registered to align with `fixed_img`.

        warped_grid : ndarray
            Image showing deformation applied to a regular grid.

        bk_dxdy : ndarray
            (2, N, M) numpy array of pixel displacements in the
            x and y directions.

        """

        if moving_xy is not None:
            moving_xy, fixed_xy = self.filter_xy(moving_xy, fixed_xy,
                                                 moving_img.shape,
                                                 mask)

        self.moving_xy = moving_xy
        self.fixed_xy = fixed_xy
        warped_img, warp_grid, bk_dxdy = \
            NonRigidRegistrar.register(self, moving_img=moving_img,
                                       fixed_img=fixed_img,
                                       mask=mask,
                                       moving_xy=moving_xy,
                                       fixed_xy=fixed_xy,
                                       **kwargs)

        return warped_img, warp_grid, bk_dxdy

    def filter_xy(self, moving_xy, fixed_xy, img_shape_rc, mask=None):
        """Remove points outside image and/or mask

        """

        if mask is None:
            mask = np.full(img_shape_rc, 255, dtype=np.uint8)

        moving_inside_idx = warp_tools.get_inside_mask_idx(moving_xy, mask)
        fixed_inside_idx = warp_tools.get_inside_mask_idx(fixed_xy, mask)
        inside_idx = np.intersect1d(moving_inside_idx, fixed_inside_idx)

        return moving_xy[inside_idx, :], fixed_xy[inside_idx, :]

# Class members that perform non-rigid registrations #
class SimpleElastixWarper(NonRigidRegistrarXY):
    """Uses SimpleElastix to register images

    May optionally using corresponding points

    """
    def __init__(self, params=None, ammi_weight=0.33,
                 bending_penalty_weight=0.33, kp_weight=0.33):
        """
        Parameters
        ----------
        ammi_weight : float
            Weight given to the AdvancedMattesMutualInformation metric.

        bending_penalty_weight : float
            Weight given to the TransformBendingEnergyPenalty metric.

        kp_weight : float
            Weight given to the CorrespondingPointsEuclideanDistanceMetric
            metric. Only used if moving_xy and fixed_xy are provided as
            arguments to the `register()` method.

        """
        super().__init__(params=params)

        self.ammi_weight = ammi_weight
        self.bending_penalty_weight = bending_penalty_weight
        self.kp_weight = kp_weight


    @staticmethod
    def get_default_params(img_shape, grid_spacing_ratio=0.025):
        """
        Get default parameters for registration with sitk.ElastixImageFilter

        See https://simpleelastix.readthedocs.io/Introduction.html
        for advice on parameter selection
        """
        p = sitk.GetDefaultParameterMap("bspline")
        p["Metric"] = ['AdvancedMattesMutualInformation', 'TransformBendingEnergyPenalty']
        p["MaximumNumberOfIterations"] = ['1500']  # Can try up to 2000
        p['FixedImagePyramid'] = ["FixedRecursiveImagePyramid"]
        p['MovingImagePyramid'] = ["MovingRecursiveImagePyramid"]
        p['Interpolator'] = ["BSplineInterpolator"]
        p["ImageSampler"] = ["RandomCoordinate"]
        p["MetricSamplingStrategy"] = ["None"]  # Use all points
        p["UseRandomSampleRegion"] = ["true"]
        p["ErodeMask"] = ["true"]
        p["NumberOfHistogramBins"] = ["32"]
        p["NumberOfSpatialSamples"] = ["3000"]
        p["NewSamplesEveryIteration"] = ["true"]
        p["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
        p["ASGDParameterEstimationMethod"] = ["DisplacementDistribution"]
        p["HowToCombineTransforms"] = ["Compose"]
        grid_spacing_x = img_shape[1]*grid_spacing_ratio
        grid_spacing_y = img_shape[0]*grid_spacing_ratio
        grid_spacing = str(int(np.mean([grid_spacing_x, grid_spacing_y])))
        p["FinalGridSpacingInPhysicalUnits"] = [grid_spacing]
        p["WriteResultImage"] = ["false"]

        p["BSplineTransformSplineOrder"] = ['3']

        return p

    @staticmethod
    def elastix_invert_transform(registed_elastix_obj, sitk_fixed):
        """Invert transformation as described in elastix manual.

        See section 6.1.6: DisplacementMagnitudePenalty: inverting transformations

        Parameters
        ----------
        registed_elastix_obj: sitk.ElastixImageFilter
            sitk.ElastixImageFilter object that has completed
            image registration.

        sitk_fixed : SimpleITK.Image
            SimpleITK.Image created from the fixed image.

        Returns
        -------
        inverted_deformationField : ndarray
            (N,M,2) numpy array of pixel displacements in the
            x and y directions.

        NOTE
        ----
        sitk.IterativeInverseDisplacementField seems to do a better job,
        and is what is used in warp_tools.get_inverse_field. However, this
        method is maintained in case one would like to use it.

        """

        inverse_transformationFilter = sitk.TransformixImageFilter()
        transf_parameter_map = registed_elastix_obj.GetTransformParameterMap()
        transf_parameter_map[0]["Metric"] = ["DisplacementMagnitudePenalty"]
        transf_parameter_map[0]["HowToCombineTransforms"] = ["Compose"]
        inverse_transformationFilter.SetMovingImage(sitk_fixed)
        inverse_transformationFilter.SetTransformParameterMap(transf_parameter_map)
        inverse_transformationFilter.ComputeDeformationFieldOn()
        inverse_transformationFilter.Execute()
        inverted_deformationField = sitk.GetArrayFromImage(inverse_transformationFilter.GetDeformationField())

        return inverted_deformationField

    def read_warpped_pts(self, fname):
        with open(fname, 'r') as f:
            lines = f.readlines()
            results = []
            for line in lines:
                warped_coords = np.array(((line.split('OutputPoint = [ ')[1]).split(' ]\t')[0]).split(' ')).astype(float)
                results.append(warped_coords)
        return np.stack(results, axis=0)

    def write_elastix_kp(self, kp, fname):
        """Temporarily write fixed_xy and moving_xy to file

        Parameters
        ----------
        kp: ndarray
            (N, 2) numpy array of points (xy).

        fname: str
            Name of file in which to save the points.

        """

        argfile = open(fname, 'w')
        npts = kp.shape[0]
        argfile.writelines(f"index\n{npts}\n")
        # argfile.writelines(f"point\n{npts}\n")
        for idx, i in enumerate(range(npts)):
            xy = kp[i]
            # argfile.writelines(f"{idx+1} {xy[0]} {xy[1]}\n") # 2
            # argfile.writelines(f"{0} {xy[0]} {xy[1]}\n") # 1
            # argfile.writelines(f"{xy[0]} {xy[1]} {0}\n") # 3 (not work)
            argfile.writelines(f"{xy[0]} {xy[1]}\n") # 1

    def run_elastix(self, moving_img, fixed_img, moving_xy=None, fixed_xy=None,
                    params=None, mask=None, src_pts=None, warped_img_save_path=None, moving_img_src=None, fixed_img_src=None):

        """Run SimpleElastix to register images.

        Can using corresponding points to aid in registration by providing
        moving_xy and fixed_xy.

        Parameters
        ----------
        moving_img : ndarray
            Image to warp to align with `fixed_img`.

        fixed_img : ndarray
            Image `moving_img` is warped to align with.

        moving_xy : ndarray, optional
            (N, 2) array containing points in the moving image that correspond
            to those in the fixed image.

        fixed_xy : ndarray, optional
            (N, 2) array containing points in the fixed image that correspond
            to those in the moving image.

        mask : ndarray, optional
            2D array with shape (N,M) where non-zero pixel values are
            foreground, and 0 is background, which is ignnored during
            registration. If None, then all non-zero pixels in images
            will be used to create the mask.

        """

        elastix_image_filter_obj = sitk.ElastixImageFilter()

        if moving_xy is not None and fixed_xy is not None:

            rand_id = np.random.randint(0, 10000)
            fixed_kp_fname = os.path.join(pathlib.Path(__file__).parent,
                                          f"{rand_id}_fixedPointSet.pts")
            moving_kp_fname = os.path.join(pathlib.Path(__file__).parent,
                                           f"{rand_id}_movingPointSet.pts")

            self.write_elastix_kp(fixed_xy, fixed_kp_fname)
            self.write_elastix_kp(moving_xy, moving_kp_fname)

            kp_dist_met = "CorrespondingPointsEuclideanDistanceMetric"
            current_metrics = list(params["Metric"])
            if not self._params_provided or kp_dist_met not in current_metrics:
                current_metrics.append(kp_dist_met)
                params["Metric"] = current_metrics
                weights = np.array([self.ammi_weight,
                                    self.bending_penalty_weight,
                                    self.kp_weight])

            elastix_image_filter_obj.SetParameterMap(params)
            elastix_image_filter_obj.SetFixedPointSetFileName(fixed_kp_fname)
            elastix_image_filter_obj.SetMovingPointSetFileName(moving_kp_fname)
        else:
            weights = np.array([self.ammi_weight, self.bending_penalty_weight])

        # Set metric weights #
        weights /= weights.sum()
        n_metrics = len(params["Metric"])
        n_res = eval(params["NumberOfResolutions"][0])
        for r in range(n_metrics):
            params[f'Metric{r}Weight'] = [str(weights[r])]*n_res

        elastix_image_filter_obj.SetParameterMap(params)

        # Perform registration #
        sitk_moving = sitk.GetImageFromArray(moving_img)
        sitk_fixed = sitk.GetImageFromArray(fixed_img)
        elastix_image_filter_obj.SetMovingImage(sitk_moving)
        elastix_image_filter_obj.SetFixedImage(sitk_fixed)

        if mask is not None:
            sitk_mask = sitk.Cast(sitk.GetImageFromArray(mask.astype(np.uint8)),
                                  sitk.sitkUInt8)

            elastix_image_filter_obj.SetFixedMask(sitk_mask)

        elastix_image_filter_obj.Execute()

        # Warp image #
        if warped_img_save_path:
            if moving_img_src is not None:
                warped_images = []
                for i in range(moving_img_src.shape[2]):
                    moving_img_slice = moving_img_src[..., i]
                    transformix = sitk.TransformixImageFilter()
                    parameter_map = elastix_image_filter_obj.GetTransformParameterMap()
                    transformix.SetTransformParameterMap(parameter_map)
                    transformix.SetMovingImage(sitk.GetImageFromArray(moving_img_slice))
                    transformix.Execute()
                    resultImage = sitk.GetArrayFromImage(transformix.GetResultImage())
                    warped_images.append(resultImage.clip(0, 255))
                warped_images = np.stack(warped_images, axis=-1)
                blend_img(warped_images.astype(np.uint8), fixed_img_src.astype(np.uint8), save_path=warped_img_save_path, alpha=0.4) # alpha operates on fixed image
            else:
                resultImage = elastix_image_filter_obj.GetResultImage()
                resultImage = sitk.GetArrayFromImage(resultImage)
                blend_img(resultImage.astype(np.uint8), fixed_img.astype(np.uint8), save_path=warped_img_save_path, alpha=0.5)

        # Warp source points (for eval):
        if src_pts is not None:

            # warp landmarks
            transformix = sitk.TransformixImageFilter()
            transformix.SetTransformParameterMap(elastix_image_filter_obj.GetTransformParameterMap())
            src_eval_pts_path = os.path.join(pathlib.Path(__file__).parent,
                                          f"src_eval.pts")
            tgt_warpped_pts_path = os.path.join(pathlib.Path(__file__).parent)

            self.write_elastix_kp(src_pts, src_eval_pts_path)
            transformix.SetFixedPointSetFileName(src_eval_pts_path)
            # The transformed points will be written to a file named outputpoints.txt
            transformix.SetOutputDirectory(tgt_warpped_pts_path)
            transformix.Execute()

            # Load warpped pts:
            warpped_pts = self.read_warpped_pts(os.path.join(pathlib.Path(__file__).parent,
                                          f"outputpoints.txt"))
        else:
            warpped_pts = None

        return warpped_pts

    def calc(self, moving_img, fixed_img, mask=None,
             moving_xy=None, fixed_xy=None, src_pts=None, warped_img_save_path=None, moving_img_src=None, fixed_img_src=None, *args, **kwargs):
        """Perform non-rigid registration using SimpleElastix.

        Can include corresponding points to help in registration by providing
        `moving_xy` and `fixed_xy`.
        """
        if not self._params_provided:
            self.params = self.get_default_params(moving_img.shape)

        warped_pts = self.run_elastix(moving_img, fixed_img,
                             moving_xy=moving_xy, fixed_xy=fixed_xy,
                             params=self.params, mask=mask, src_pts=src_pts, warped_img_save_path=warped_img_save_path, moving_img_src=moving_img_src, fixed_img_src=fixed_img_src)
        return warped_pts