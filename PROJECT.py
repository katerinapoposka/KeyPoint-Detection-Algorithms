import cv2
import numpy as np
import matplotlib.pyplot as plt


def matching_SURF_SIFT(im1, im2, d, d_r, k, k_r):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(d, d_r)
    matches = sorted(matches, key=lambda x: x.distance)
    # draw first 50 matches
    match_img = cv2.drawMatches(im1, k, im2, k_r, matches[:30], None)
    cv2.imshow('Matches', match_img)
    cv2.waitKey()


def test_image(img):
    test_image = cv2.pyrDown(img)
    test_image = cv2.pyrDown(test_image)
    num_rows, num_cols = test_image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), 30, 1)
    test_image = cv2.warpAffine(test_image, rotation_matrix, (num_cols, num_rows))
    test_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
    test_img_copy = test_image.copy()
    return test_gray, test_img_copy


def matching_orb(im1, im2, d, d_r, k, k_r):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d, d_r)
    matches = sorted(matches, key=lambda x: x.distance)
    # drawing first 50 matches
    match_img = cv2.drawMatches(im1, k, im2, k_r, matches[:50], None)
    cv2.imshow('Matches', match_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def MSER(path):
    # Create MSER object
    mser = cv2.MSER_create()

    img = cv2.imread(path)
    # Converting to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_copy = img.copy()
    # detecting regions in gray scale image
    regions, boxes = mser.detectRegions(gray)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(image_copy, hulls, 1, (0, 255, 0))

    # Creating test image by adding Scale Invariance and Rotational Invariance
    test_gray, test_img_copy = test_image(img)
    regions2, boxes2 = mser.detectRegions(test_gray)
    hulls2 = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions2]
    cv2.polylines(test_img_copy, hulls2, 1, (0, 255, 0))

    cv2.imshow('MSER1', image_copy)
    cv2.imshow('MSER2', test_img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def HARRIS(path):
    # Reading in the image
    image = cv2.imread(path)
    # Making a copy of the image
    image_copy = np.copy(image)
    # Converting to grayscale
    gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    # Detecting corners
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # Dilating corner image to enhance corner points
    dst = cv2.dilate(dst, None)

    # Thresholding paramenter, can be different for other images
    thresh = 0.1 * dst.max()
    # Creating an image copy to draw corners on
    corner_image = np.copy(image_copy)

    # Iterating through all the corners and drawing them on the image (if they pass the threshold)
    for j in range(0, dst.shape[0]):
        for i in range(0, dst.shape[1]):
            if dst[j, i] > thresh:
                # image, center pt, radius, color, thickness
                cv2.circle(corner_image, (i, j), 1, (0, 255, 0), 1)

    # Creating test image by adding Scale Invariance and Rotational Invariance
    test_gray, test_img_copy = test_image(image)
    dst2 = cv2.cornerHarris(test_gray, 2, 3, 0.04)
    dst2 = cv2.dilate(dst, None)
    thresh2 = 0.1 * dst2.max()
    for j in range(0, dst2.shape[0]):
        for i in range(0, dst2.shape[1]):
            if dst[j, i] > thresh2:
                # image, center pt, radius, color, thickness
                cv2.circle(test_img_copy, (i, j), 1, (0, 255, 0), 1)
    cv2.imshow("HARRIS1", corner_image)
    cv2.imshow("HARRIS2", test_img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def shithomasi(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # specify the number of corners, quality level, minimum eucledian distance
    corners = cv2.goodFeaturesToTrack(gray, 25, 0.1, 10)
    corners = np.int0(corners)
    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)

    # Creating test image by adding scale invariance and rotation invariance
    test_gray, test_copy = test_image(img)
    corners2 = cv2.goodFeaturesToTrack(gray, 10, 0.1, 10)
    corners2 = np.int0(corners)
    for i in corners2:
        x, y = i.ravel()
        cv2.circle(test_copy, (x, y), 3, 255, -1)

    cv2.imshow('SHITHOMASI1', img)
    cv2.imshow('SHITHOMASI2', test_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def SIFT(path1, path2):
    img1 = cv2.imread(path1, 0)
    img2 = cv2.imread(path2, 0)

    sift = cv2.xfeatures2d.SIFT_create()

    # finding the keypoints and descriptors with SIFT
    (kps1, descs1) = sift.detectAndCompute(img1, None)
    (kps2, descs2) = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descs1, descs2, k=2)

    # Finding the 2 nearest neighbors for a given descriptor. d1/d2 ratio should be smaller than a given threshold to be accepted.

    # Applying ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kps1, img2, kps2, good, None, flags=2)
    plt.imshow(img3), plt.show()
    cv2.imwrite('img3.jpg', img3)


def BRIEF(path):
    # Loading the image
    image = cv2.imread(path)
    # Converting the training image to gray scale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Creating test image by adding Scale Invariance and Rotational Invariance
    test_gray, test_img_copy = test_image(image)

    # Detecting keypoints and Create Descriptor
    fast = cv2.FastFeatureDetector_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    train_keypoints = fast.detect(img_gray, None)
    test_keypoints = fast.detect(test_gray, None)

    train_keypoints, train_descriptor = brief.compute(img_gray, train_keypoints)
    test_keypoints, test_descriptor = brief.compute(test_gray, test_keypoints)

    keypoints_without_size = np.copy(img_gray)
    keypoints_with_size = np.copy(img_gray)

    cv2.drawKeypoints(img_gray, train_keypoints, keypoints_without_size, color=(0, 255, 0))

    cv2.drawKeypoints(img_gray, train_keypoints, keypoints_with_size,
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Displaying image with and without keypoints size
    fx, plots = plt.subplots(1, 2, figsize=(20, 10))

    plots[0].set_title("Train keypoints With Size")
    plots[0].imshow(keypoints_with_size, cmap='gray')

    plots[1].set_title("Train keypoints Without Size")
    plots[1].imshow(keypoints_without_size, cmap='gray')

    # Matching Keypoints

    # Creating a Brute Force Matcher object.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Performing the matching between the BRIEF descriptors of the training image and the test image
    matches = bf.match(train_descriptor, test_descriptor)

    # The matches with shorter distance are the ones we want.
    matches = sorted(matches, key=lambda x: x.distance)

    result = cv2.drawMatches(img_gray, train_keypoints, test_gray, test_keypoints, matches, test_gray, flags=2)

    # Displaing the best matching points
    plt.title('Best Matching Points')
    plt.imshow(result)
    plt.show()

    # Printing total number of matching points between the training and query images
    print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(matches))


def SUSAN(path):
    img_color = cv2.imread(path)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    # creating mask of 37 pixels
    mask = np.ones((7, 7))
    mask[0, 0] = 0
    mask[0, 1] = 0
    mask[0, 5] = 0
    mask[0, 6] = 0
    mask[1, 0] = 0
    mask[1, 6] = 0
    mask[5, 0] = 0
    mask[5, 6] = 0
    mask[6, 0] = 0
    mask[6, 1] = 0
    mask[6, 5] = 0
    mask[6, 6] = 0
    img = img.astype(np.float64)
    thresh = 37 / 2

    output = np.zeros(img.shape)

    for i in range(3, img.shape[0] - 3):
        for j in range(3, img.shape[1] - 3):
            ir = np.array(img[i - 3:i + 4, j - 3:j + 4])
            ir = ir[mask == 1]
            ir0 = img[i, j]
            a = np.sum(np.exp(-((ir - ir0) / 10) ** 6))
            if a <= thresh:
                a = thresh - a
            else:
                a = 0
            output[i, j] = a

    finaloutput1 = img_color.copy()
    finaloutput1[output != 0] = [0, 255, 0]
    cv2.imshow("SUSAN1", finaloutput1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def DoG(path):
    img = cv2.imread(path, 0)
    low_sigma = cv2.GaussianBlur(img, (0, 0), 1, borderType=cv2.BORDER_REPLICATE)
    high_sigma = cv2.GaussianBlur(img, (0, 0), 3, borderType=cv2.BORDER_REPLICATE)
    img_dog = (high_sigma - low_sigma)
    # normalizing by the largest absolute value so range is -1 to
    img_dog = img_dog / np.amax(np.abs(img_dog))
    img_dog2 = (255.0 * (0.5 * img_dog + 0.5)).clip(0, 255).astype(np.uint8)
    # Calculating the DoG by subtracting
    dog = low_sigma - high_sigma
    cv2.imshow("low sigma", low_sigma)
    cv2.waitKey()
    cv2.imshow("high sigma", high_sigma)
    cv2.waitKey()
    cv2.imshow("DoG", img_dog2)
    cv2.waitKey()
    cv2.destroyAllWindows()


def LoG(path):
    img = cv2.imread(path, 0)

    # Applying Gaussian Blur
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    # Applying Laplacian operator in some higher datatype
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    # But this tends to localize the edge towards the brighter side.
    laplacian1 = laplacian / laplacian.max()
    cv2.imshow('laplacian', laplacian1)
    cv2.waitKey(0)

    image = laplacian
    z_c_image = np.zeros(image.shape)
    # For each pixel, count the number of positive
    # and negative pixels in the neighborhood
    regions = []
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            negative_count = 0
            positive_count = 0
            neighbour = [image[i + 1, j - 1], image[i + 1, j], image[i + 1, j + 1], image[i, j - 1],
                         image[i, j + 1], image[i - 1, j - 1], image[i - 1, j], image[i - 1, j + 1]]
            cor = [[i, j], [i + 1, j - 1], [i + 1, j], [i + 1, j + 1], [i, j - 1],
                   [i, j + 1], [i - 1, j - 1], [i - 1, j], [i - 1, j + 1]]
            d = max(neighbour)
            e = min(neighbour)
            for h in neighbour:
                if h > 0:
                    positive_count += 1
                elif h < 0:
                    negative_count += 1
            # If both negative and positive values exist in
            # the pixel neighborhood, then that pixel is a
            # potential zero crossing
            z_c = ((negative_count > 0) and (positive_count > 0))
            # Change the pixel value with the maximum neighborhood
            # difference with the pixel
            if z_c:
                if image[i, j] > 0:
                    z_c_image[i, j] = image[i, j] + np.abs(e)
                elif image[i, j] < 0:
                    z_c_image[i, j] = np.abs(image[i, j]) + d
                regions.append(cor)
    # Normalize and change datatype to 'uint8' (optional)
    z_c_norm = z_c_image / z_c_image.max() * 255
    z_c_image = np.uint8(z_c_norm)
    new_image = z_c_image
    cv2.imshow("zero-crossings", new_image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def FAST(path):
    i = cv2.imread(path)
    img = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    # Initiating FAST object with default values
    fast = cv2.FastFeatureDetector_create()
    # finding and drawing the keypoints
    keypoints = fast.detect(img, None)
    img2 = cv2.drawKeypoints(i, keypoints, None, color=(255, 0, 0))

    test_gray, test_img_copy = test_image(i)
    keypoints2 = fast.detect(test_gray, None)
    img3 = cv2.drawKeypoints(test_img_copy, keypoints, None, color=(255, 0, 0))

    cv2.imshow('FAST1', img2)
    cv2.imshow('FAST2', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ORB(path):
    i = cv2.imread(path)
    img = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    # Initiating ORB detector
    orb = cv2.ORB_create()
    # finding the keypoints with ORB
    keypoints = orb.detect(img, None)
    # computing the descriptors with ORB
    keypoints, des = orb.compute(img, keypoints)
    # drawing only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(i, keypoints, None, color=(0, 255, 0), flags=0)
    # rotated
    t, rotated = test_image(i)
    keypoints_r = orb.detect(rotated, None)
    keypoints_r, des_r = orb.compute(rotated, keypoints_r)
    img3 = cv2.drawKeypoints(rotated, keypoints_r, None, color=(0, 255, 0), flags=0)
    cv2.imshow("ORB", img2)
    cv2.waitKey()
    cv2.imshow("ORB2", img3)
    cv2.waitKey()
    cv2.destroyAllWindows()
    matching_orb(i, rotated, des, des_r, keypoints, keypoints_r)


def SURF():
    i = cv2.imread('ZGRADA.jpg')
    img = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create()
    keypoints, des = surf.detectAndCompute(img, None)
    kp_img = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0),
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # rotated
    t, rotated = test_image(i)
    rotated_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    keypoints_r, des_r = surf.detectAndCompute(rotated_gray, None)
    kp_img_r = cv2.drawKeypoints(rotated, keypoints_r, None, color=(0, 255, 0),
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('SURF', kp_img)
    cv2.waitKey()
    cv2.imshow("SURF rotated", kp_img_r)
    cv2.waitKey()
    matching_SURF_SIFT(i, rotated, des, des_r, keypoints, keypoints_r)


def SalientRegions(path):
    i = cv2.imread(path)
    img = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    # initialising OpenCV’s static saliency spectral residual detector and
    # compute the saliency map
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(img)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    # Applying thresholding to get binary image from saliency map
    threshMap = cv2.threshold(saliencyMap, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("original", img)
    cv2.waitKey()
    cv2.imshow("saliency map", saliencyMap)
    cv2.waitKey()
    cv2.imshow("thresholding image", threshMap)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    SUSAN('ZGRADI.jpg'),
    ORB('ZGRADI.jpg'),
    MSER('ZGRADI.jpg'), HARRIS('ZGRADI.jpg'), SUSAN('ZGRADI.jpg'), DoG('ZGRADI.jpg'), LoG('ZGRADI.jpg'), shithomasi(
        'ZGRADI.jpg'), FAST('ZGRADI.jpg')
