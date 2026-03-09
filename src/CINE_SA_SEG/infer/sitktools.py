import os.path
import re
import uuid

import numpy as np
import SimpleITK as sitk


__version__ = "0.9.6"

_dicomTagLUT = {
    "studyDate": "0008|0020",
    "seriesDate": "0008|0021",
    "contentDate": "0008|0022",
    "contentTime": "0008|0023",
    "acquisitionDateTime": "0008|002a",
    "studyTime": "0008|0030",
    "seriesTime": "0008|0031",
    "accessionNumber": "0008|0050",
    "modality": "0008|0060",
    "manufacturer": "0008|0070",
    "institutionName": "0008|0080",
    "studyDescription": "0008|1030",
    "seriesDescription": "0008|103e",
    "patientName": "0010|0010",
    "patientID": "0010|0020",
    "patientBirthDate": "0010|0030",
    "patientSex": "0010|0040",
    "patientAge": "0010|1010",
    "bodyPartExamined": "0018|0015",
    "studyInstanceUID": "0020|000d",
    "frameOfReferenceUID": "0020|0052",
    "seriesInstanceUID": "0020|000e",
    "seriesNumber": "0020|0011",
    "windowCenter": "0028|1050",
    "windowWidth": "0028|1051",
    "rescaleSlope": "0028|1053",
    "rescaleIntercept": "0028|1052",
    "sopClassUID": "0008|0016",
}


def labelOverlay(im, mask, opacity=0.5, window=None):
    """Overlay label in RGB over given image.

    :param im: sitk.Image.
    :param mask: sitk.Image.
    :param opacity: Opacity of label mask, float [0, 1], default=0.5.
    :param window: None or (wc, ww), perform IntensityWindowing on im if provided.
    :return: RGB VectorUInt8 image with label overlayed.
    """

    maskRGB = sitk.GetArrayFromImage(sitk.LabelToRGB(mask))
    maskAlpha = sitk.GetArrayFromImage(mask > 0)
    maskAlpha = 1 - (maskAlpha * opacity)
    maskAlpha = maskAlpha[..., np.newaxis]
    if window is None:
        imRescaled = sitk.GetArrayFromImage(sitk.RescaleIntensity(im, 0, 255))
    else:
        imRescaled = sitk.GetArrayFromImage(
            sitk.IntensityWindowing(im, window[0] - window[1] // 2, window[0] + window[1] // 2, 0, 255)
        )
    if im.GetNumberOfComponentsPerPixel() == 1:
        imRescaled = imRescaled[..., np.newaxis]
    resIm = sitk.GetImageFromArray((imRescaled * maskAlpha + maskRGB * (1 - maskAlpha)).astype("uint8"), isVector=True)
    resIm.CopyInformation(im)
    return resIm


def resampleByRef(
    im,
    ref=None,
    transform=None,
    spacing=None,
    size=None,
    interpolator=sitk.sitkLinear,
    defaultValue=None,
    pixelID=None,
):
    """Smart image resampler. Resample im according to ref at optionally
    provided spacing and size.

    :param im: source image.
    :param ref: reference image. Origin and direction will be copied from the reference. None = im.
    :param transform: sitk.Transform that will be performed before resampling.
    :param spacing: Output image spacing that will be calculated automatically if not provided.
    :param size: Output image size that will be calculated automatically if not provided.
    :param interpolator: default=sitk.sitkLinear.
    :param defaultValue: default=minimal intensity of input image.
    :param pixelID: default=the same as ref.
    :return: Resampled image.
    """
    if ref is None:
        ref = im
    if spacing is None and size is None:
        size = np.array(ref.GetSize())
        spacing = np.array(ref.GetSpacing())
    elif size is None:
        spacing = np.array(spacing)
        size = np.round((np.array(ref.GetSize()) * ref.GetSpacing() / spacing)).astype("int")
    else:
        size = np.array(size).astype("int")
        spacing = np.array(ref.GetSize()) * ref.GetSpacing() / size
    if transform is None:
        if len(im.GetSize()) == 3:
            transform = sitk.Euler3DTransform()
        else:
            transform = sitk.Euler2DTransform()
    if defaultValue is None:
        defaultValue = float(np.min(sitk.GetArrayViewFromImage(im)))
    if pixelID is None:
        pixelID = ref.GetPixelID()
    return sitk.Resample(
        im,
        size.tolist(),
        transform,
        interpolator,
        ref.GetOrigin(),
        spacing.tolist(),
        ref.GetDirection(),
        defaultValue,
        pixelID,
    )


def loadDicom(dicomDir, returnTags=False):
    """SimpleITK dicom reader.

    :param dicomDir: list of filenames or path to dicom files.
    :param returnTags: return dicom tags of each dicom file. default to False.
    :return: Image or (image, tags) if returnTags==True.
             tags: [{'slice': sliceNo,
                     'fileName': fileName,
                     'tags': {dicomTag: value}}, ...]
    """
    reader = sitk.ImageSeriesReader()
    if type(dicomDir) is list:
        dicom_names = dicomDir
    elif os.path.isdir(dicomDir):
        dicom_names = reader.GetGDCMSeriesFileNames(dicomDir)
    else:
        raise ValueError("Unrecognized dicomDir parameter: %s" % str(dicomDir))
    if returnTags:
        reader.SetMetaDataDictionaryArrayUpdate(True)
    reader.SetFileNames(dicom_names)
    im = reader.Execute()
    if returnTags:
        tags = [
            {
                "slice": i,
                "fileName": dicom_names[i],
                "tags": {k: reader.GetMetaData(i, k) for k in reader.GetMetaDataKeys(i)},
            }
            for i in range(im.GetDepth())
        ]
        return im, tags
    else:
        return im


def generateImFromBox(refIm, bndBoxes, edgesOnly=False, label=1):
    """Draw bounding box in image.

    :param refIm: reference Image (2D).
    :param bndBoxes: List of bounding boxes. [[w_min, w_max, h_min, h_max], ...]
    :param edgesOnly: Draw only edges. default to False.
    :param label: Label number of boxes. default to 1.
    :return: SimpleITK label image of boxes.
    """
    assert len(refIm.GetSize()) == 2
    resIm = np.zeros(refIm.GetSize()[::-1], np.uint8)
    for curBndbox in bndBoxes:
        if not edgesOnly:
            resIm[curBndbox[2] : (curBndbox[3] + 1), curBndbox[0] : (curBndbox[1] + 1)] = label
        else:
            resIm[curBndbox[2] : (curBndbox[3] + 1), max(curBndbox[0], 0)] = label
            resIm[curBndbox[2] : (curBndbox[3] + 1), min(curBndbox[1], resIm.shape[1] - 1)] = label
            resIm[max(curBndbox[2], 0), curBndbox[0] : (curBndbox[1] + 1)] = label
            resIm[min(curBndbox[3], resIm.shape[0] - 1), curBndbox[0] : (curBndbox[1] + 1)] = label
    resImSitk = sitk.GetImageFromArray(resIm)
    resImSitk.CopyInformation(refIm)
    return resImSitk


def cropRotatedRect(im, rotatedRect, indexPos=True, interpolator=sitk.sitkLinear):
    """Crop a rotated rect region from a 2D image.

    :param im: reference
    :param rotatedRect: [center_w, center_h, dim_w, dim_h, angle], angle(degree) to vector [1, 0], clockwise
    :param indexPos: Center & dim are expressed in indical coordinates. False = physical coordinates. default to True.
    :param interpolator: default to sitk.sitkLinear.
    :return: Rect patch. sitk.Image (2d).
    """
    assert len(rotatedRect) == 5
    assert im.GetDimension() == 2
    rotatedRectNp = np.array(rotatedRect)
    angle = rotatedRectNp[-1] * np.pi / 180.0

    direction = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    spacing = np.array(im.GetSpacing())
    if indexPos:
        centerPoint = np.array(im.TransformContinuousIndexToPhysicalPoint(rotatedRectNp[:2].tolist()))
    else:
        centerPoint = rotatedRectNp[:2]
    origin = (
        centerPoint
        - direction[0] * spacing[0] * rotatedRectNp[2] / 2
        - direction[1] * spacing[1] * rotatedRectNp[3] / 2
    )
    resIm = sitk.Resample(
        im,
        rotatedRectNp[2:4].astype("int").tolist(),
        sitk.AffineTransform(2),
        interpolator,
        origin.tolist(),
        spacing.tolist(),
        direction.T.ravel().tolist(),
    )
    return resIm


def normalized(a, axis=-1, order=2):
    """Normalize numpy vectors with norm by axis.

    :param a: np.ndarray.
    :param axis: axis to perform norm. default to -1.
    :param order: order of norm, default to 2.
    :return: Normalized vectors. np.ndarray.
    """
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def resliceImageByCenter(
    im, centerPos, directionW, directionH, width=10, spacing=(0.1, 0.1), thickness=1, interpolator=sitk.sitkLinear
):
    """Reslice a square patch from 3d SimpleITK image by patch center and
    directions.

    :param im: sitk.Image. Must be 3d.
    :param centerPos: Center of output patch. [w, h, d] in physical coordinates.
    :param directionW: Width direction vector of output image, will be normalized. [w, h, d]. ||directionW|| != 0.
    :param directionH: Height direction vector of output image, will be normalized. [w, h, d]. ||directionH|| != 0.
    :param width: Width of output square patch in physical coordinates (mm). default to 10(mm).
    :param spacing: Spacing of output square patch. default to (0.1, 0.1).
    :param thickness: Thickness of output square patch. default to 1(mm).
    :param interpolator: default to sitk.sitkLinear.
    :return: square patch as indicated. sitk.Image (2d)
    """
    assert im.GetDimension() == 3
    directionW = normalized(np.array(directionW)).ravel()
    directionH = normalized(np.array(directionH)).ravel()
    spacing = np.array(spacing)
    centerPos = np.array(centerPos)

    if not np.abs(np.dot(directionW, directionH)) < 1e-4:
        raise ValueError("directionW and directionH should be perpendicular.")
    directionN = np.cross(directionH, directionW)
    size = np.round(width / spacing).astype("int").tolist()
    origin = centerPos - directionW * width / 2 - directionH * width / 2
    return sitk.Resample(
        im,
        size + [1],
        sitk.AffineTransform(3),
        interpolator,
        origin.tolist(),
        spacing.tolist() + [thickness],
        np.stack([directionW, directionH, directionN]).T.ravel().tolist(),
    )[:, :, 0]


def rotationMatrix(axis, theta):
    """Return the rotation matrix associated with counterclockwise rotation
    about the given axis by theta radians.

    :param axis: rotation axis, 3d np.array.
    :param theta: theta in radians, counterclockwise.
    :return: Rotation matrix. 3*3 np.array.
    """
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def getDirectionFromNormal(normal, targetDirW=(0, 1, 0)):
    """Get directionW and directionH of plane from normal vector of a plane and
    a target width direction.

    :param normal: Normal vector of plane. 3d vector in np.array.
    :param targetDirW: Target width direction vector.
                       Projection of this vector in the plane will be used as directionW.
    :return: (directionW, directionH)
    """
    targetDirW = normalized(np.array(targetDirW)).ravel()
    normal = normalized(normal).ravel()
    newDirW = normalized(targetDirW - np.dot(normal, targetDirW) * normal).ravel()
    newDirH = np.cross(normal, newDirW)
    return (newDirW, newDirH)


def simpleRegistration(
    fixed,
    moving,
    fixedMask=None,
    movingMask=None,
    transform=None,
    initialize=True,
    iterations=100,
    lr=0.1,
    samplingPercentage=0.1,
    histogramBins=25,
    multiLevel=None,
):
    """Simple simpleitk based registration with mattes MI metric and gradient
    descent optimizer.

    :param fixed: Fixed Image.
    :param moving: Moving Image.
    :param fixedMask: Fixed mask. default to None.
    :param movingMask: Moving mask. default to None.
    :param transform: Initial moving to fixed transform. default to None = AffineTransform.
    :param initialize: Initialize transform by sitk.CenteredTransformInitializer. default to True.
    :param iterations: Iterations, default to 100.
    :param lr: Learning rate, default to 0.1.
    :param samplingPercentage: Sampling percentage to mattes MI metric. default to 0.1.
    :param histogramBins: Histogram bins to calculate mattes MI metric. default to 25.
    :param multiLevel: Multilevel shrink & smooth, None: disabled,
                           if an int (k) is provided: shrink factor: [2**k for k in range(k)], smooth sigma: range(k),
                           if a tuple is provided: ([shrink factor], [smooth sigma]).
                           default to None.
    :return: moving to fixed transform.
    """
    if transform is None:
        transform = sitk.AffineTransform(fixed.GetDimension())
    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)
    if initialize:
        initial_transform = sitk.CenteredTransformInitializer(
            fixed, moving, transform, sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    else:
        initial_transform = transform

    registration_method = sitk.ImageRegistrationMethod()
    if fixedMask is not None:
        registration_method.SetMetricFixedMask(fixedMask)
    if movingMask is not None:
        registration_method.SetMetricMovingMask(movingMask)
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=histogramBins)
    registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
    registration_method.SetMetricSamplingPercentage(samplingPercentage)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=lr, numberOfIterations=iterations)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    if multiLevel is not None:
        if type(multiLevel) is int:
            shrinkFactor = [2**k for k in range(multiLevel)]
            smoothFactor = [k for k in range(multiLevel)]
        elif type(multiLevel) is tuple:
            shrinkFactor = multiLevel[0]
            smoothFactor = multiLevel[1]
        else:
            raise ValueError("Unrecognized multiLevel parameter: %s" % str(multiLevel))
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=shrinkFactor)
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothFactor)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    translation = registration_method.Execute(fixed, moving)
    return translation


def getBoundingBox(mask, label=1):
    """Get the bounding box of a label mask.

    :param mask: sitk.Image
    :param label: label id, default=1.
    """
    labStat = sitk.LabelShapeStatisticsImageFilter()
    labStat.Execute(mask)
    return labStat.GetBoundingBox(label)


def cubeCrop(im, imMask, targetShape=256, padding=10, pixelID=None, defaultValue=None, interpolator=sitk.sitkLinear):
    """Crop an image into a isotrophic cube with respect to the center of
    bounding box of imMask.

    :param im: sitk.Image (3d),
    :param imMask: sitk.Image.
    :param targetShape: Target indical width of the cube, default=256.
    :param padding: Padding width around imMask bounding box in physical coord(mm), default=10(mm).
    :param defaultValue: default=minimal intensity of input image.
    :param pixelID: default=the same as im.
    """
    assert im.GetDimension() == 3

    bndbox = np.array(getBoundingBox(imMask)).astype("float")
    spacing = np.array(imMask.GetSpacing())
    center = bndbox[:3] + bndbox[3:] / 2  # indical pos

    dim = bndbox[3:] * spacing  # suppose det(imMask_direction) = 1, hopefully this is true
    maxDim = np.max(dim)
    cropDim = maxDim + padding  # physical coord.

    origin = center - (cropDim / 2) / spacing  # indical coord
    newSpacing = cropDim / targetShape
    if pixelID is None:
        pixelID = im.GetPixelID()
    if defaultValue is None:
        defaultValue = float(np.min(sitk.GetArrayViewFromImage(im)))
    resIm = sitk.Resample(
        im,
        [int(targetShape)] * 3,
        sitk.Euler3DTransform(),
        interpolator,
        imMask.TransformContinuousIndexToPhysicalPoint(origin),
        [float(newSpacing)] * 3,
        im.GetDirection(),
        defaultValue,
        pixelID,
    )
    return resIm


def transposeNormalizeCoordinate(origin, direction, spacing, size):
    """Transpose oriented bounding box coordinates into LPS, [W, H, D] form.

    :param origin: origin, 3D vector
    :param direction: direction, 3*3 matrix. Note: direction matrix should be in [\vec w^T, \vec h^T, \vec d^T].
                      Specifically,
                      1) if you use direction from sitk.Image, just pass im.GetDirection(),
                      2) if you use direction from GetOrientedBoundingBoxDirection(), pass np.array(GetOrientedBoundingBoxDirection()).reshape([3, 3]).T
    :param spacing: spacing, 3D vector
    :param size: size, 3D vector
    :return: (transposedOrigin, transposedDirMat, transposedSpacing, transposedSize)
    """
    origin = np.array(origin).reshape([3])
    dirMat = np.array(direction).reshape([3, 3])
    size = np.array(size).reshape([3])
    spacing = np.array(spacing).reshape([3])
    maxIdx = np.abs(dirMat).argmax(axis=0)
    sgn = np.sign(dirMat[maxIdx, np.arange(3)])

    transposedDirMat = dirMat[:, maxIdx.argsort()] * sgn[maxIdx.argsort()]

    transposedSpacing = spacing[maxIdx.argsort()]
    transposedSize = size[maxIdx.argsort()]
    transposedOriginIdx = np.zeros(3)
    transposedOriginIdx[sgn < 0] = size[sgn < 0] - 1
    transposedOrigin = np.matmul(np.matmul(dirMat, np.diag(spacing)), transposedOriginIdx) + origin
    return transposedOrigin, transposedDirMat, transposedSpacing, transposedSize


def transposeNormalize(im):
    """Transpose image such that sitk.GetArrayFromImage(im) can be standard [D,
    H, W] array with D increasing foot to head, H superior to posterior and W
    right to left.

    :param im: sitk.Image (3d)
    :return: sitk.Image
    """
    dirMat = np.array(im.GetDirection()).reshape([3, 3])
    spacing = np.array(im.GetSpacing())
    size = np.array(im.GetSize())
    origin = np.array(im.GetOrigin())
    transposedOrigin, transposedDirMat, transposedSpacing, transposedSize = transposeNormalizeCoordinate(
        origin, dirMat, spacing, size
    )
    transposedIm = sitk.Resample(
        im,
        transposedSize.tolist(),
        sitk.Euler3DTransform(),
        sitk.sitkNearestNeighbor,
        transposedOrigin.tolist(),
        transposedSpacing.tolist(),
        transposedDirMat.ravel().tolist(),
    )
    return transposedIm


def composeImageIntoRGB(
    fixed, moving, fixedWindow=None, movingWindow=None, fixedColor=(190, 20, 190), interpolator=sitk.sitkLinear
):
    """Compose two image into RGB image to evaluate registration performance.

    :param fixed: fixedImage
    :param moving: movingImage, will be automatically resampled into fixed image space.
    :param windowFixed: Intensity window for fixed image. None: Rescale into 0-255. default to None.
    :param windowMoving: Intensity window for moving image. None: Rescale into 0-255. default to None.
    :param fixedColor: Color in uint8 RGB for fixed image, moving image will be 255 - fixedColor.
    """
    if fixedWindow is None:
        w1 = sitk.RescaleIntensity(fixed, 0, 255)
    else:
        w1 = sitk.IntensityWindowing(
            fixed, fixedWindow[0] - fixedWindow[1] // 2, fixedWindow[0] + fixedWindow[1] // 2, 0, 255
        )
    moving = sitk.Resample(moving, fixed, sitk.AffineTransform(fixed.GetDimension()), interpolator)
    if movingWindow is None:
        w2 = sitk.RescaleIntensity(moving, 0, 255)
    else:
        w2 = sitk.IntensityWindowing(
            moving, movingWindow[0] - movingWindow[1] // 2, movingWindow[0] + movingWindow[1] // 2, 0, 255
        )
    w1Np = sitk.GetArrayFromImage(w1)
    w2Np = sitk.GetArrayFromImage(w2)

    fixedColor = np.array(fixedColor).astype("uint8").astype("float") / 255
    movingColor = 1 - fixedColor
    resIm = sitk.GetImageFromArray(
        (w1Np[..., np.newaxis] * fixedColor + w2Np[..., np.newaxis] * movingColor).astype("uint8"), isVector=True
    )
    resIm.CopyInformation(w1)
    return resIm


def dicomUIDGenerator(orgRoot="1.2.826.0.1.3680043.10.398", length=48):
    if not orgRoot[-1] == ".":
        orgRoot += "."
    return orgRoot + str(uuid.uuid4().int)[: (length - len(orgRoot))]


def _isSITKDicomTag(s):
    return re.match(r"^[\dA-Fa-f]{4}\|[\dA-Fa-f]{4}$", s) is not None


def writeDicomSeries(
    im,
    outputPrefix,
    reverse=False,
    floatWidth=3,
    compress=False,
    studyInstanceUID=None,
    seriesInstanceUID=None,
    seriesNumber=1,
    frameOfReferenceUID=None,
    modality="",
    patientName="",
    patientSex="",
    patientID="",
    seriesDescription="",
    bodyPartExamined="",
    accessionNumber="",
    manufacturer="",
    studyTime="",
    studyDate="",
    seriesTime="",
    sopClassUID="1.2.840.10008.5.1.4.1.1.2",  # CT
    dicomTags=None,
    **kwargs,
):
    """Dicom series writer.

    :param im: sitk.Image.
    :param outputPrefix: prefix for output dicom files. Each image file will be suffixed with '%.4d.dcm' % instanceNumber.
    :param reverse: Reverse the depth axis when generating dicoms. May be necessary if handling chest CT.
    :param floatWidth: Floats (origin, direction, spacing) are saved in fixed-width to prevent rounding issues.
    :param compress: SimpleITK.ImageFileWriter based compression. default=False.
    :param studyInstanceUID, seriesInstanceUID, frameOfReferenceUID: UIDs will be generated automatically if not provided.
    :param dicomTags: dicom tags can be provided as dict of simpleitk compliant tag string (such as '02fa|02a1') as key.
    :param ...: Several dicom tag names can be recognized automatically, refer to dicomTagLUT dict for more.
    """
    assert im.GetDimension() == 3

    # generate uids
    if studyInstanceUID is None:
        studyInstanceUID = dicomUIDGenerator()
    if seriesInstanceUID is None:
        seriesInstanceUID = dicomUIDGenerator()
    if frameOfReferenceUID is None:
        frameOfReferenceUID = dicomUIDGenerator()

    # generate tags
    _tags = {}
    for k, v in locals().items():
        if k in _dicomTagLUT.keys():
            _tags[_dicomTagLUT[k].lower()] = v

    for k, v in kwargs.items():
        if k in _dicomTagLUT.keys():
            _tags[_dicomTagLUT[k].lower()] = v
    # simpleITK dicom writer has a default SOP class uid of Secondary Capture Image Storage
    # which renders gdcm-based readers (including sitk.ImageSeriesReader) unable to properly calculate depth-spacing
    # use MR or CT instead

    if dicomTags is not None:
        for k, v in dicomTags.items():
            if _isSITKDicomTag(k):
                _tags[k.lower()] = v
    # float width string
    floatString = "%%.%df" % floatWidth
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    depList = list(range(im.GetDepth()))
    if reverse:
        depList.reverse()

    imDir = np.array(im.GetDirection()).reshape(3, 3).T.ravel()  # simpleitk directions are expressed in column-major

    for instanceNumber, dep in enumerate(depList):
        curImSlice = im[:, :, dep]

        curTags = _tags.copy()
        curTags["0020|0013"] = instanceNumber + 1  # instance number
        curTags["0020|0032"] = "\\".join([floatString % t for t in im.TransformIndexToPhysicalPoint((0, 0, dep))])
        curTags["0020|0037"] = "\\".join([floatString % t for t in imDir[:6]])
        curTags["0028|0030"] = "\\".join([floatString % t for t in im.GetSpacing()[:2]])

        for k, v in curTags.items():
            if type(v) is list:
                curImSlice.SetMetaData(k, "\\".join(["%s" % t for t in v]))
            else:
                curImSlice.SetMetaData(k, str(v))

        writer.SetFileName("%s%.4d.dcm" % (outputPrefix, curTags["0020|0013"]))
        writer.SetUseCompression(compress)
        writer.Execute(curImSlice)
    return _tags


def convIndexToPhysicalPoint(ind, origin, spacing, direction):
    """Convert array of 3d indical points to physical points.

    :param ind: [n * 3], should be in [[w, h, d], [w, h, d] ... ]
    :param origin: [3]
    :param spacing: [3]
    :param direction: [9] use the direction array directly from im.GetDirection()
    :return: [3 * n] physical point array, in [[w, w, ...], [h, h, ...], [d, d, ...]]
    """
    assert ind.shape[1] == 3
    direction_ = np.array(direction, dtype=np.float_).reshape(3, 3)
    origin_ = np.array(origin, dtype=np.float_).reshape(3, 1)
    spacing_ = np.diag(spacing)
    return np.matmul(np.matmul(direction_, spacing_), ind.T) + origin_


def convPhysicalPointToIndex(phy, origin, spacing, direction):
    """Convert array of 3d physical points to indical points.

    :param ind: [n * 3], should be in [[w, h, d], [w, h, d] ... ]
    :param origin: [3]
    :param spacing: [3]
    :param direction: [9] use the direction array directly from im.GetDirection()
    :return: [3 * n] *float* indical point array, in [[w, w, ...], [h, h, ...], [d, d, ...]],
    """
    assert phy.shape[1] == 3
    direction_ = np.array(direction, dtype=np.float_).reshape(3, 3)
    origin_ = np.array(origin, dtype=np.float_).reshape(3, 1)
    spacing_ = np.diag(spacing)
    return np.matmul(np.linalg.pinv(np.matmul(direction_, spacing_)), (phy - origin_.T).T)


def cropByBoundingBox(im, bndbox=None, indicalPos=True, padding=None):
    """Crop an image by bounding box.

    :param im: sitk image.
    :param bndbox: 3d bndbox should be [w_start, h_start, d_start, w_width, h_width, d_width]
                   as returned by getBoundingBox(im).
    :param indicalPos: coordinates are indical. default=True.
    :param padding: None or [w, h, d] / [w, h]. will be padded symetrically to both sides. default = None
    """
    if not indicalPos:
        raise NotImplementedError("Physical position bndbox is not implemented yet.")
    if bndbox is None:
        bndbox = getBoundingBox(im)
    if padding is None:
        padding = [0, 0, 0]
    elif not hasattr(padding, "__iter__"):
        padding = [padding] * 3

    if len(bndbox) == 6:
        return im[
            max(0, bndbox[0] - padding[0]) : min(im.GetWidth(), bndbox[0] + bndbox[3] + padding[0]),
            max(0, bndbox[1] - padding[1]) : min(im.GetHeight(), bndbox[1] + bndbox[4] + padding[1]),
            max(0, bndbox[2] - padding[2]) : min(im.GetDepth(), bndbox[2] + bndbox[5] + padding[2]),
        ]
    elif len(bndbox) == 4:
        return im[
            max(0, bndbox[0] - padding[0]) : min(im.GetWidth(), bndbox[0] + bndbox[2] + padding[0]),
            max(0, bndbox[1] - padding[1]) : min(im.GetHeight(), bndbox[1] + bndbox[3] + padding[1]),
        ]
    else:
        raise ValueError("unrecognized dimensions of bndbox. %s" % str(bndbox))


def _getAllCorners(im):
    """Get all corner points of the image.

    :param im: sitk image
    :return cornersPhysicalCoord, cornersIndicalCoord:
    """
    dim = im.GetDimension()
    size = np.array(im.GetSize())
    cornersInd = np.array(
        [
            [0 if _j == "0" else size[_k] for _k, _j in enumerate(np.binary_repr(_i, width=dim))]
            for _i in np.arange(2**dim)
        ]
    )
    return convIndexToPhysicalPoint(cornersInd, im.GetOrigin(), im.GetSpacing(), im.GetDirection()).T, cornersInd


def coerceImage(
    args, spacing=None, interpolator=sitk.sitkLinear, defaultValue=None, coercePixelID=False, pixelID=None
):
    """Coerce sitk images by largest common physical area so that you can
    easily perform multi-image operations.

    :param args: list of sitk images
    :param spacing: target spacing, will be minimum of all image in each dimension if not provided
    :param interpolator: default=sitk.sitkLinear
    :param defaultValue: None=min of all input images. default=None.
    :param coercePixelID: make the pixel id of all output images the same. default=False
    :return: [im, im, im...]
    """
    if len(args) == 0:
        return None
    dim = args[0].GetDimension()
    minCorner = np.array([np.inf] * dim)
    maxCorner = np.array([-np.inf] * dim)
    if spacing is None:
        spacing = np.array([np.inf] * dim)
        calculateSpacing = True
    else:
        calculateSpacing = False
    if defaultValue is None:
        calculateDefaultValue = True
        defaultValue = np.inf
    else:
        calculateDefaultValue = False
    if pixelID is None:
        calculatePixelID = True
        newPixelID = 0
    else:
        calculatePixelID = False
    pixelIDs = []

    for i in args:
        assert type(i) is sitk.Image
        assert i.GetDimension() == dim
        phy, _ = _getAllCorners(i)
        minCorner = np.minimum(minCorner, phy.min(axis=0))
        maxCorner = np.maximum(maxCorner, phy.max(axis=0))
        if calculateSpacing:
            spacing = np.minimum(spacing, i.GetSpacing())
        if calculateDefaultValue:
            defaultValue = np.min(defaultValue, np.min(sitk.GetArrayViewFromImage(i)))
        if not coercePixelID:
            pixelIDs.append(i.GetPixelID())
        elif calculatePixelID:
            newPixelID = max(newPixelID, i.GetPixelID())
        else:
            pixelIDs.append(pixelID)

    if calculatePixelID:
        pixelIDs = [newPixelID] * len(args)

    size = (maxCorner - minCorner) / spacing
    return [
        sitk.Resample(
            i,
            size.astype("int").tolist(),
            sitk.AffineTransform(dim),
            interpolator,
            minCorner.tolist(),
            spacing.tolist(),
            np.diag(np.ones(dim)).ravel().tolist(),
            float(defaultValue),
            pixelID,
        )
        for i, pixelID in zip(args, pixelIDs)
    ]


def fastLabelGaussian(
    imSeg,
    ref=None,
    spacing=None,
    original=False,
    radius=5,
    thres=0.5,
    interpolator=sitk.sitkLinear,
    overrideLabelwiseFilters=None,
):
    """Fast label gaussian for interpolating multilabel images with smoothing
    while preserving boundary. Inspired by original ITK implementation of
    LabelImageGaussianInterpolator and itkGenericLabelInterpolator.

    :param imSeg: sitk.Image, label image.
    :param ref: Reference image. Either ref or spacing must be provided.
    :param spacing: Target spacing. Either ref or spacing must be provided.
    :param original: Use sitk.sitkLabelGaussian as interpolator. Much slower but proven.
                     Other params will be ignored if set. default = False.
    :param radius: Kernel radius for IIR gaussian. default = 5.
    :param interpolator: Base interpolator. default = sitk.sitkLinear.
    :param thres: Threshold for output mask. default = 0.5.
    :param overrideLabelwiseFilters: Override the list of filters to perform iterratively after interpolation of each label.
                                     default = [lambda x: sitk.RecursiveGaussian(x, radius)]
    :return: imSeg interpolated.
    """
    if ref is None:
        ref = imSeg
    if spacing is None:
        spacing = ref.GetSpacing()
    if overrideLabelwiseFilters is None:
        overrideLabelwiseFilters = [lambda x: sitk.RecursiveGaussian(x, radius)]

    boxPadding = radius

    if original:
        # use original sitk.sitkLabelGaussian interpolation in cropped image
        cropped = cropByBoundingBox(imSeg, getBoundingBox(imSeg > 0), padding=boxPadding)
        cropped = resampleByRef(cropped, spacing=spacing, interpolator=sitk.sitkLabelGaussian, defaultValue=0)
        interp_crop = resampleByRef(
            cropped,
            ref,
            spacing=spacing,
            defaultValue=0,
            interpolator=sitk.sitkNearestNeighbor,
            pixelID=sitk.sitkUInt8,
        )
        return interp_crop

    labStat = sitk.LabelShapeStatisticsImageFilter()
    labStat.Execute(imSeg)

    allLabs = []

    imSegRef = resampleByRef(
        imSeg, ref, spacing=spacing, interpolator=sitk.sitkNearestNeighbor, defaultValue=0, pixelID=sitk.sitkUInt8
    )
    maxBox = cropByBoundingBox(imSegRef, getBoundingBox(imSegRef > 0), padding=boxPadding)

    for lab in labStat.GetLabels():
        curLabel = cropByBoundingBox(imSeg, labStat.GetBoundingBox(lab), padding=boxPadding) == lab
        interp = resampleByRef(
            curLabel, spacing=spacing, interpolator=interpolator, defaultValue=0, pixelID=sitk.sitkFloat32
        )
        filtered = interp
        for labelwiseFilter in overrideLabelwiseFilters:
            filtered = labelwiseFilter(filtered)
        labImOrg = resampleByRef(
            filtered, maxBox, spacing=spacing, interpolator=interpolator, defaultValue=0, pixelID=sitk.sitkFloat32
        )
        allLabs.append(labImOrg)

    if len(allLabs) > 1:
        sumOfLabs = sitk.NaryAdd(allLabs)
        labelMask = sumOfLabs > thres

        # SimpleITK does not have a vector argmax filter so we have to implement it by numpy
        # compose into vector image is faster than np.stack, delta~0.2s
        composeIm = sitk.Compose(allLabs)
        # manually release these intermediate float32 image
        for labIm in allLabs:
            del labIm
        composeImNp = sitk.GetArrayFromImage(composeIm)
        maxValLab = composeImNp.argmax(axis=-1).astype("uint8") + 1
        maxValLab = sitk.GetImageFromArray(maxValLab)
        maxValLab.CopyInformation(maxBox)

        maxValLab = sitk.Mask(maxValLab, labelMask)
        resIm = sitk.ChangeLabel(maxValLab, {(k + 1): lab for k, lab in enumerate(labStat.GetLabels())})
    else:
        sumOfLabs = allLabs[0]
        labelMask = sumOfLabs > thres
        resIm = sitk.ChangeLabel(labelMask, {(k + 1): lab for k, lab in enumerate(labStat.GetLabels())})
    resIm = resampleByRef(
        resIm, ref, spacing=spacing, interpolator=sitk.sitkNearestNeighbor, defaultValue=0, pixelID=sitk.sitkUInt8
    )

    return resIm


def buildChangeLabelMapping(maxLabel, keep=None, ignore=None, keepOriginalLabel=False, foreground=1, background=0):
    """Utility to create a mapping for sitk.ChangeLabelImageFilter.

    :param maxLabel: Max label id.
    :param keep: List of label ids or dict of label mapping to keep. You should set either keep or ignore.
    :param ignore: List of label ids to ignore. You should set either keep or ignore.
    :param keepOriginalLabel: Reset all kept labels to foreground if False, or keep original label if True.
    :param foreground: Foreground label id.
    :param background: Background label id.
    """
    if keep is None and ignore is None:
        raise ValueError("Set either keep or ignore, not none.")
    if keep is not None and ignore is not None:
        raise ValueError("Set either keep or ignore, not both.")
    if keep is not None:
        if keepOriginalLabel:
            if type(keep) is list:
                return {lab: lab if lab in keep else background for lab in range(1, maxLabel + 1)}
            elif type(keep) is dict:
                return {lab: keep[lab] if lab in keep else background for lab in range(1, maxLabel + 1)}
        else:
            return {lab: foreground if lab in keep else background for lab in range(1, maxLabel + 1)}
    else:
        if keepOriginalLabel:
            return {lab: background if lab in ignore else lab for lab in range(1, maxLabel + 1)}
        else:
            return {lab: background if lab in ignore else foreground for lab in range(1, maxLabel + 1)}


def filterLabel(imSeg, keep=None, ignore=None, keepOriginalLabel=False, foreground=1, background=0):
    """Filter label in a label image.

    :param maxLabel: Max label id.
    :param keep: List of label ids or dict of label mapping to keep. You should set either keep or ignore.
    :param ignore: List of label ids to ignore. You should set either keep or ignore.
    :param keepOriginalLabel: Reset all kept labels to foreground if False, or keep original label if True.
    :param foreground: Foreground label id.
    :param background: Background label id.
    """
    maxLabel = np.max(sitk.GetArrayFromImage(imSeg))
    return sitk.ChangeLabel(
        imSeg, buildChangeLabelMapping(maxLabel, keep, ignore, keepOriginalLabel, foreground, background)
    )


def getStraightenSliceProps(centerline, targetShape=(96, 96), outputSpacing=None):
    y = centerline["y"]
    B = centerline["B"]
    N = centerline["N"]
    T = centerline["T"]
    lSpacing = centerline["lSpacing"]
    if outputSpacing is None:
        outputSpacing = (lSpacing, lSpacing)
    orgs = y - B * targetShape[1] * outputSpacing[1] / 2 - N * targetShape[0] * outputSpacing[0] / 2
    dirs = np.concatenate([B, N, T], axis=1).reshape([-1, 3, 3]).transpose([0, 2, 1])
    spacing = np.diag((outputSpacing[0], outputSpacing[1], lSpacing))
    return orgs, dirs, spacing


def getStraightenGrid(centerline, targetShape=(96, 96), outputSpacing=None, displacementField=False):
    orgs, drcs, spc = getStraightenSliceProps(centerline, targetShape=targetShape, outputSpacing=outputSpacing)
    sliceIndPos = np.repeat(
        np.stack(np.meshgrid(np.arange(targetShape[0]), np.arange(targetShape[1]), 0, indexing="ij")).T,
        orgs.shape[0],
        axis=0,
    )  # w, h, d
    slicePhyPos = orgs.reshape([orgs.shape[0], 1, 1, 3]) + np.einsum(
        "dwhc,dce->dwhe", sliceIndPos, (drcs @ spc).transpose([0, 2, 1])
    )

    if displacementField:
        # assume displacement field org=(0, 0, 0), spacing=(1, 1, 1)
        transformGridMesh = (
            np.stack(
                np.meshgrid(
                    np.arange(slicePhyPos.shape[0]),
                    np.arange(slicePhyPos.shape[1]),
                    np.arange(slicePhyPos.shape[2]),
                    indexing="ij",
                )[::-1]
            )
            .transpose([1, 2, 3, 0])
            .astype("float")
        )  # [D， H， W, vec3], vec3 in [W,H,D]
        dispF = slicePhyPos - transformGridMesh
        return dispF
    else:
        return slicePhyPos


def _straightenCPR_ind2phy_nearest(orgs, drcs, spc, ind_pos):
    slice_idx = np.clip(np.round(ind_pos[:, 2]).astype("int"), 0, orgs.shape[0] - 1)
    return orgs[slice_idx] + np.einsum(
        "da,dab->db",
        np.concatenate([ind_pos[:, :2], np.zeros([ind_pos.shape[0], 1], dtype=ind_pos.dtype)], axis=1),
        (drcs @ spc).transpose([0, 2, 1])[slice_idx],
    )


def _straightenCPR_ind2phy_linear(orgs, drcs, spc, ind_pos):
    ind_pos_ceil = ind_pos.copy()
    ind_pos_floor = ind_pos.copy()
    ind_pos_floor[:, 2] = np.floor(ind_pos[:, 2])
    ind_pos_ceil[:, 2] = ind_pos_floor[:, 2] + 1
    mix_ratio_floor = 1 - (ind_pos[:, 2] - ind_pos_floor[:, 2])
    mix_ratio_ceil = 1 - (ind_pos_ceil[:, 2] - ind_pos[:, 2])
    return mix_ratio_floor[:, np.newaxis] * _straightenCPR_ind2phy_nearest(
        orgs, drcs, spc, ind_pos_floor
    ) + mix_ratio_ceil[:, np.newaxis] * _straightenCPR_ind2phy_nearest(orgs, drcs, spc, ind_pos_ceil)


def straightenSliceInd2Phy(ind_pos, centerline, interpolator="nearest", targetShape=(96, 96), outputSpacing=None):
    """Convert indical positions in straightened slices to physical position.

    Args:
        ind_pos (np.ndarray): Indical pos ([[W, H, D], ..])
        centerline (Dict[str, np.ndarray]): Centerline, output of st.stableFrenet.
        interpolator (str, optional): Interpolation between slices, linear or nearest. Defaults to 'nearest'.
        targetShape (tuple, optional): Target shape. Defaults to (96, 96).
        outputSpacing (tuple, optional): Output spacing. Defaults to None.

    Returns:
        np.ndarray: physical position
    """
    if len(ind_pos.shape) == 1:
        ind_pos = ind_pos[np.newaxis]
    orgs, drcs, spc = getStraightenSliceProps(centerline, targetShape, outputSpacing)
    if interpolator == "nearest":
        return _straightenCPR_ind2phy_nearest(orgs, drcs, spc, ind_pos)
    elif interpolator == "linear":
        return _straightenCPR_ind2phy_linear(orgs, drcs, spc, ind_pos)


def straightenCPR(
    im, centerline, targetShape=(96, 96), outputSpacing=None, interpolator=sitk.sitkLinear, defaultValue=0
):
    """Straighten curved planar reformation on given image and centerline.
    Build centerline object with stableFrenet function.

    :param im: sitk.Image
    :param centerline: Output of stableFrenet function.
    :param targetShape: Canvas size, default to (128, 128).
    :param outputSpacing: Spacing of axial image, default to lSpacing.
    :param interpolator: Interpolator, default = sitk.sitkLinear.
    :param defaultValue: Default value for outside the canvas voxels. default=0.
    :return: Stacked axial image as straighten CPR image volume. Size = [targetShape[0], targetShape[1], len(centerline['y'])]
             If you want medial image, use resliceImageByCenter to reslice it along [0, 0, 1] direction
    """
    lSpacing = centerline["lSpacing"]
    if outputSpacing is None:
        outputSpacing = (lSpacing, lSpacing)
    grid = getStraightenGrid(centerline, targetShape=targetShape, outputSpacing=outputSpacing, displacementField=True)
    gridIm = sitk.GetImageFromArray(grid, isVector=True)
    gridImF = sitk.DisplacementFieldTransform(gridIm)
    resIm = sitk.Resample(
        im,
        grid.shape[:-1][::-1],
        gridImF,
        interpolator,
        (0, 0, 0),  # displacement field generated this way
        (1, 1, 1),
        np.eye(3).ravel(),
        defaultValue,
    )
    resIm.SetSpacing((outputSpacing[0], outputSpacing[1], lSpacing))
    return resIm


def stretchedCPR(
    im,
    centerline,
    targetWidth=512,
    initPos=256,
    angle=0.0,
    outputSpacing=None,
    interpolator=sitk.sitkLinear,
    defaultValue=0,
):
    """Stretched curved planar reformation on given image and centerline. Build
    centerline object with stableFrenet function.

    :param im: sitk.Image
    :param centerline: Output of stableFrenet function.
    :param targetWidth: Canvas width, default to 512.
    :param initPos: Starting point (indical point [initPos, 0]) for vessel reformation in output image. default = 256 ( 512 // 2).
    :param angle: Reformation angle to N_0 in degrees. default = 0.
    :param outputSpacing: Spacing in width direction of output image.
    :param interpolator: Interpolator, default = sitk.sitkLinear.
    :param defaultValue: Default value for outside the canvas voxels. default=0.
    :return: Stretched CPR image. ( Spacing: [outputSpacing, lSpacing] )
    """
    y = centerline["y"]
    B = centerline["B"]
    lSpacing = centerline["lSpacing"]
    if outputSpacing is None:
        outputSpacing = lSpacing
    spacing = np.array([outputSpacing, outputSpacing, lSpacing], dtype="float")
    extend = np.array([targetWidth, 1, 1], dtype="int")
    imRes = []
    L = np.array([np.cos(angle / 180.0 * np.pi), np.sin(angle / 180.0 * np.pi), 0], dtype="float")
    initDist = initPos * spacing[0]
    for i in range(y.shape[0]):
        curDir = np.concatenate([L, -np.cross(L, B[i]), B[i]]).reshape([3, 3]).T.ravel()
        curOrg = y[i] - L * (initDist + np.dot(L, y[i] - y[0]))
        curIm = sitk.Resample(
            im,
            extend.astype("int").tolist(),
            sitk.AffineTransform(3),
            interpolator,
            curOrg.tolist(),
            spacing.tolist(),
            curDir.tolist(),
            defaultValue,
        )
        imRes.append(sitk.GetArrayFromImage(curIm).reshape([extend[1], extend[0]]))
    resIm = sitk.GetImageFromArray(np.concatenate(imRes))
    resIm.SetSpacing([outputSpacing, lSpacing])
    return resIm


def getStraightenSlice(strIm, angle=0, interpolator=sitk.sitkLinear, defaultValue=0):
    """Generate rotated slice along medial axis in straightened CPR.

    :param strIm: Straightened CPR Image. Output of straightenCPR.
    :param angle: Viewing angle ( in degrees ). Default = 0.
    :param interpolator: Interpolator, default = sitk.sitkLinear.
    :param defaultValue: Default value for outside the canvas voxels. default=0.
    :return: Single slice of straightened CPR in specified angle.
    """
    dirMat = rotationMatrix([0, 0, 1], angle / 180.0 * np.pi)
    dirMat = dirMat.T[[2, 0, 1]]
    origin = strIm.TransformContinuousIndexToPhysicalPoint(
        np.array([strIm.GetWidth() / 2, strIm.GetWidth() / 2, 0]) - dirMat[1] * strIm.GetWidth() / 2
    )
    curAngleSlice = sitk.Resample(
        strIm,
        [strIm.GetDepth(), strIm.GetWidth(), 1],
        sitk.Euler3DTransform(),
        interpolator,
        origin,
        [strIm.GetSpacing()[-1], strIm.GetSpacing()[0], strIm.GetSpacing()[1]],
        dirMat.T.ravel(),
        defaultValue,
    )
    return curAngleSlice[:, :, 0]


def stableFrenet(vessel, T=None, keepOriginalDim=False):
    """Build centerline frenet frame system with respect to centerline point.
    Input data must be arclen parameterized.

    :param vessel: Physical position of centerline point. np.array ([N * 3]).
    :param T: Externally calculated tangent of vessel. Optional. np.array ([N * 3]).
    :param keepOriginalDim: Make sure the number of points in the vessel does not change after calculation.
        Otherwise `output.shape[0] = vessel.shape[0] - 1`. Default to False.
    :return: centerline object with y, N, T, B and lSpacing.
    """

    # refer to http://www.unchainedgeometry.com/jbloom/pdf/ref-frames.pdf
    # proposed by Ken Sloan

    d1 = np.diff(vessel, axis=0)
    lSpacing = np.mean(np.linalg.norm(d1, axis=1))
    lSpacingStd = np.std(np.linalg.norm(d1, axis=1))

    if lSpacingStd / lSpacing > 0.25:
        raise ValueError(
            "lSpacing does not equal. Vessel points is not arclen parameterized? %.5f, %.5f, %.5f"
            % (lSpacing, lSpacingStd, lSpacingStd / lSpacing)
        )
    if T is not None:
        assert T.shape == vessel.shape
        Torg = T
        if not keepOriginalDim:
            Torg = Torg[:-1]
    else:
        Torg = d1 / lSpacing
        if keepOriginalDim:
            Torg = np.r_[Torg, Torg[-2:-1]]
    N0 = normalized(Torg[1] - Torg[0])[0]
    N1 = normalized(Torg[2] - Torg[1])[0]
    B0 = normalized(N1 - N0)[0]
    Ns = [N0]
    Bs = [B0]
    for i in range(1, Torg.shape[0]):
        T1 = Torg[i]
        N1 = np.cross(B0, T1)
        B1 = np.cross(T1, N1)
        Ns.append(normalized(N1)[0])
        Bs.append(normalized(B1)[0])
        N0 = N1
        B0 = B1
    if keepOriginalDim:
        return {"y": vessel, "T": Torg, "N": np.stack(Ns), "B": np.stack(Bs), "lSpacing": lSpacing}
    else:
        return {"y": vessel[:-1], "T": Torg, "N": np.stack(Ns), "B": np.stack(Bs), "lSpacing": lSpacing}


def connectedComponent(im, fullyConnected=False):
    """Connected component image filter for polyfilling a bug in connected
    component image filter in SimpleITK==1.2.2 that leads to segfault when last
    dimensions of image equals 1.

    :param im:  sitk.Image, input image.
    :param fullyConnected: bool, is fully connected.
    :return: sitk.Image, connected components
    """
    size = im.GetSize()
    if (len(size) == 2 and size[1] == 1) or (len(size) == 3 and size[0] > 2 and size[1] == 1 and size[2] == 1):
        imOrg = im
        im = sitk.GetImageFromArray(sitk.GetArrayFromImage(im).reshape([size[0], 1]))
        compNp = sitk.GetArrayFromImage(sitk.ConnectedComponent(im, fullyConnected))
        resIm = sitk.GetImageFromArray(compNp.reshape(size[::-1]))
        resIm.CopyInformation(imOrg)
        return resIm
    return sitk.ConnectedComponent(im, fullyConnected)


def relabelConnectedComponent(im, minimumObjectSize=None, returnObjectSize=False):
    """Relabel connected component sorted by physical size in descending order.

    :param im: sitk.Image
    :param minimumObjectSize: Remove objects smaller than minimumObjectSize (in voxels).
    :param returnObjectSize: Return voxel volume of each relabeled object.
    :returns: sitk.Image if returnObjectSize==False; (sitk.Image, Tuple[int]) if returnObjectSize
    """
    relab = sitk.RelabelComponentImageFilter()
    if minimumObjectSize is not None:
        relab.SetMinimumObjectSize(minimumObjectSize)
    res = relab.Execute(connectedComponent(im))
    if returnObjectSize:
        return res, relab.GetSizeOfObjectsInPixels()
    else:
        return res


def getOrientedBoundingBox(labStat, label):
    """Get oriented bounding box from sitk.LabelShapeStatisticsImageFilter.

    :param labStat: instance of sitk.LabelShapeStatisticsImageFilter
    :param label: label
    :return: origin, direction, size
    """
    origin = np.array(labStat.GetOrientedBoundingBoxOrigin(label))
    direction = np.array(labStat.GetOrientedBoundingBoxDirection(label)).reshape([3, 3]).T
    spacing = [1, 1, 1]
    size = np.array(labStat.GetOrientedBoundingBoxSize(label))
    origin, direction, _, size = transposeNormalizeCoordinate(origin, direction, spacing, size)
    return origin, direction, size


def getOrientedBoundingBoxVertices(origin, direction, size):
    """Get coordinates of all 8 vertices of oriented bounding box.

    :param origin, direction, size: output of getOrientedBoundingBox
    :return: 8*3 array.
    """
    cornerDir = np.array([[(i & (1 << j)) >> j for j in range(3)] for i in range(2**3)])
    return np.matmul(direction, (cornerDir * size).T).T + origin


def distanceToLine(points, linePnt1, linePnt2):
    """Return distances of points to a line defined by two points.

    :param points: n*3 array.
    :param linePnt1, linePnt2: two R^3 vectors that are on the line.
    :return: distances
    """
    # |(linePnt1 - pnt) \cross (linePnt2 - linePnt1)| / | linePnt2 - linePnt1 |
    return np.linalg.norm(np.cross(linePnt1 - points, linePnt2 - linePnt1), axis=1) / np.linalg.norm(
        linePnt2 - linePnt1
    )


def anglesBetween(vec, dirs, posDir=None):
    """Returns the signed angle between a vector (vec) and vectors (dirs).

    :param vec: 3d vector.
    :param dirs: n*3 vector or 3d vector.
    :param posDir: positive direction. will be calculated as vec x dirs[0] if not provided.
    :return: signed angle in radians.
    """
    if len(dirs.shape) == 1:
        dirs = dirs[np.newaxis]
    if posDir is None:
        posDir = normalized(np.cross(vec, dirs[0]))[0]
    sinVal = np.dot(np.cross(vec, dirs), posDir)
    cosVal = np.dot(dirs, vec)
    angles = np.arccos(cosVal)
    angles[np.sign(sinVal) < 0] = 2 * np.pi - angles[np.sign(sinVal) < 0]
    return angles


def arclen(x, ord=2):
    """Returns arclen as calculated by cumulated sum of distances between
    consecutive points.

    :param x: n*m vector indicating coordinates of n points in m-D.
    :param ord: order of norm, default to 2.
    :return: n-D vector, arclen.
    """
    return np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(x, axis=0), axis=1, ord=ord))])


def generatePadImage(imNp, dim=3, patch_size=(96, 96, 96), step=(64, 64, 64)):
    """Generate a padded image with array of starting position suitable for
    patch-based pipeline.

    :param imNp: np.ndarray, image array. len(imNp.shape) >= dim.
    :param dim: image dimension. will be calculated from last dimension in imNp.
                e.g. dim=3, imNp: [NCDHW] or [NDHW] or [DHW].
    :param patch_size: size of patch.
    :param step: step between patches.
    :return: imPad, pos. padded image and starting position of each patch.
    """
    assert dim == len(patch_size) and dim == len(step)
    assert dim <= len(imNp.shape)
    org_shape = np.array(imNp.shape[-dim:])
    pad_shape = (np.ceil((org_shape - patch_size) / step) * step + patch_size - org_shape).astype("int")
    imPad = np.pad(imNp, [(0, 0)] * (len(imNp.shape) - dim) + [(0, pad_shape[i]) for i in range(dim)])
    final_shape = np.array(imPad.shape[-dim:])
    pos = (
        np.stack(np.meshgrid(*[np.arange(0, final_shape[i] - patch_size[i] + 1, step[i]) for i in range(dim)]))
        .reshape([dim, -1])
        .T
    )
    return imPad, pos


def thresholdLevelSet(
    im,
    init_seg,
    lower,
    upper,
    curvatureScaling=1.0,
    propagationScaling=1.0,
    numberOfIterations=100,
    maxErr=0.02,
    threshold=0,
):
    lsFilter = sitk.ThresholdSegmentationLevelSetImageFilter()
    lsFilter.SetLowerThreshold(lower)
    lsFilter.SetUpperThreshold(upper)
    lsFilter.SetMaximumRMSError(maxErr)
    lsFilter.SetNumberOfIterations(numberOfIterations)
    lsFilter.SetCurvatureScaling(curvatureScaling)
    lsFilter.SetPropagationScaling(propagationScaling)
    lsFilter.ReverseExpansionDirectionOn()
    init_ls = sitk.SignedMaurerDistanceMap(init_seg > 0, useImageSpacing=True, insideIsPositive=True)
    if im.GetPixelID() != init_ls.GetPixelID():
        im = sitk.Cast(im, init_ls.GetPixelID())
    ls = lsFilter.Execute(init_ls, im)
    return ls > threshold


def gacLevelSet(
    speed_map, init_seg, curvatureScaling=1.0, propagationScaling=1.0, numberOfIterations=100, maxErr=0.02, threshold=0
):
    lsFilter = sitk.GeodesicActiveContourLevelSetImageFilter()
    lsFilter.SetMaximumRMSError(maxErr)
    lsFilter.SetNumberOfIterations(numberOfIterations)
    lsFilter.SetCurvatureScaling(curvatureScaling)
    lsFilter.SetPropagationScaling(propagationScaling)
    init_ls = sitk.SignedMaurerDistanceMap(init_seg > 0, useImageSpacing=True, insideIsPositive=True)
    if speed_map.GetPixelID() != init_ls.GetPixelID():
        speed_map = sitk.Cast(speed_map, init_ls.GetPixelID())

    ls = lsFilter.Execute(init_ls, speed_map)
    return ls > threshold


def laplacianLevelSet(
    im, init_seg, curvatureScaling=1.0, propagationScaling=1.0, numberOfIterations=100, maxErr=0.02, threshold=0
):
    lsFilter = sitk.LaplacianSegmentationLevelSet()
    lsFilter.SetMaximumRMSError(maxErr)
    lsFilter.SetNumberOfIterations(numberOfIterations)
    lsFilter.SetCurvatureScaling(curvatureScaling)
    lsFilter.SetPropagationScaling(propagationScaling)
    lsFilter.ReverseExpansionDirectionOn()
    init_ls = sitk.SignedMaurerDistanceMap(init_seg > 0, useImageSpacing=True, insideIsPositive=True)
    if im.GetPixelID() != init_ls.GetPixelID():
        im = sitk.Cast(im, init_ls.GetPixelID())
    ls = lsFilter.Execute(init_ls, im)
    return ls > threshold


def shapeDetectionLevelSet(
    speed_map, init_seg, curvatureScaling=1.0, propagationScaling=1.0, numberOfIterations=100, maxErr=0.02, threshold=0
):
    lsFilter = sitk.ShapeDetectionLevelSetImageFilter()
    lsFilter.SetMaximumRMSError(maxErr)
    lsFilter.SetNumberOfIterations(numberOfIterations)
    lsFilter.SetCurvatureScaling(curvatureScaling)
    lsFilter.SetPropagationScaling(propagationScaling)
    init_ls = sitk.SignedMaurerDistanceMap(init_seg > 0, useImageSpacing=True, insideIsPositive=True)
    if speed_map.GetPixelID() != init_ls.GetPixelID():
        speed_map = sitk.Cast(speed_map, init_ls.GetPixelID())

    ls = lsFilter.Execute(init_ls, speed_map)
    return ls > threshold


def decomposeAffine(aff, fromRAS=False):
    """Decompose an 3D affine matrix [4*4] into origin, spacing and direction.

    Args:
        aff (np.ndarray): affine matrix

    Returns:
        np.ndarray [3], np.ndarray[3], np.ndarray[3*3]:
            Origin, Spacing, Direction
    """
    if fromRAS:
        aff = np.diag([-1, -1, 1, 1]) @ aff
    O = aff[:3, 3]  # noqa: E741
    DS = aff[:3, :3]
    S = np.linalg.norm(DS, axis=0)
    D = DS @ np.diag(1 / S)
    return O, S, D


def composeAffine(origin, spacing, direction, toRAS=False):
    """Compose origin, spacing and direction into 3D affine matrix [4*4].

    Args:
        origin, spacing, direction (np.ndarray)

    Returns:
        np.ndarray [4*4]: 3D affine matrix.
    """
    O = np.array(origin)  # noqa: E741
    S = np.diag(spacing)
    D = np.array(direction).reshape([3, 3])
    aff = np.vstack([np.hstack([D @ S, O.reshape([3, 1])]), [0, 0, 0, 1]])
    if toRAS:
        return np.diag([-1, -1, 1, 1]) @ aff
    else:
        return aff


def probe(physicalPoints, im, defaultValue=0):
    """Probe an simpleitk image with list of physical points.

    Args:
        physicalPoints (np.ndarray[N*3]): Position of input physical points
        seg (sitk.Image): SimpleITK image
        defaultValue (int or float, optional): Defaults to 0.

    Returns:
        np.ndarray: probed result
    """
    orgSegNp = sitk.GetArrayViewFromImage(im)
    cntIdx = np.round(convPhysicalPointToIndex(physicalPoints, im.GetOrigin(), im.GetSpacing(), im.GetDirection()))[
        ::-1
    ].T.astype("int")
    res = np.full(physicalPoints.shape[0], defaultValue, dtype=orgSegNp.dtype)
    validMask = ~np.any((cntIdx < 0) | (cntIdx >= orgSegNp.shape), axis=1)
    res[validMask] = orgSegNp[cntIdx[validMask, 0], cntIdx[validMask, 1], cntIdx[validMask, 2]]
    return res
