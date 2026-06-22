import numpy as np
from skimage.feature import graycomatrix, graycoprops


def extract_features(img):

    features = []

    mean = np.mean(img)
    std = np.std(img)
    variance = np.var(img)

    features.extend([mean, std, variance])

    glcm = graycomatrix(img, [1], [0], 256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    features.extend([contrast, correlation, energy, homogeneity])

    return features
