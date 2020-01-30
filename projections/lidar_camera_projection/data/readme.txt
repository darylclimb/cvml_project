Data Format Description
=======================

The data for training and testing can be found in the corresponding folders.
The sub-folders are structured as follows:

  - image_02/ contains the left color camera images (png)
  - label_02/ contains the left color camera label files (plain text files)
  - calib/ contains the calibration for all four cameras (plain text file)

The label files contain the following information, which can be read and
written using the matlab tools (readLabels.m, writeLabels.m) provided within
this devkit. All values (numerical or strings) are separated via spaces,
each row corresponds to one object. The 15 columns represent:

#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.

Here, 'DontCare' labels denote regions in which objects have not been labeled,
for example because they have been too far away from the laser scanner. To
prevent such objects from being counted as false positives our evaluation
script will ignore objects detected in don't care regions of the test set.
You can use the don't care labels in the training set to avoid that your object
detector is harvesting hard negatives from those areas, in case you consider
non-object regions from the training images as negative examples.

The coordinates in the camera coordinate system can be projected in the image
by using the 3x4 projection matrix in the calib folder, where for the left
color camera for which the images are provided, P2 must be used. The
difference between rotation_y and alpha is, that rotation_y is directly
given in camera coordinates, while alpha also considers the vector from the
camera center to the object center, to compute the relative orientation of
the object with respect to the camera. For example, a car which is facing
along the X-axis of the camera coordinate system corresponds to rotation_y=0,
no matter where it is located in the X/Z plane (bird's eye view), while
alpha is zero only, when this object is located along the Z-axis of the
camera. When moving the car away from the Z-axis, the observation angle
will change.

To project a point from Velodyne coordinates into the left color image,
you can use this formula: x = P2 * R0_rect * Tr_velo_to_cam * y
For the right color image: x = P3 * R0_rect * Tr_velo_to_cam * y

Note: All matrices are stored row-major, i.e., the first values correspond
to the first row. R0_rect contains a 3x3 matrix which you need to extend to
a 4x4 matrix by adding a 1 as the bottom-right element and 0's elsewhere.
Tr_xxx is a 3x4 matrix (R|t), which you need to extend to a 4x4 matrix
in the same way!

Note, that while all this information is available for the training data,
only the data which is actually needed for the particular benchmark must
be provided to the evaluation server. However, all 15 values must be provided
at all times, with the unused ones set to their default values (=invalid) as
specified in writeLabels.m. Additionally a 16'th value must be provided
with a floating value of the score for a particular detection, where higher
indicates higher confidence in the detection. The range of your scores will
be automatically determined by our evaluation server, you don't have to
normalize it, but it should be roughly linear. If you use writeLabels.m for
writing your results, this function will take care of storing all required
data correctly.
