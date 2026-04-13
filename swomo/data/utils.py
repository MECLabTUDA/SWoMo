import torch
import numpy as np
from PIL import ImageColor

CLASSES = {
    0: {
        "name": "Background",
        "color": '#000000'
    },
    1: {
        "name": "Cornea",
        "color": '#AF3235'
    },
    2: {
        "name": "Pupil",
        "color": '#DFE674'
    },
    3: {
        "name": "Lens",
        "color": '#AF72B0'
    },
    4: {
        "name": "Slit Knife",
        "color": '#46C5DD'
    },
    5: {
        "name": "Gauge",
        "color": '#F282B4'
    },
    6: {
        "name": "Capsulorhexis Cystotome",
        "color": '#98CC70'
    },
    7: {
        "name": "Spatula",
        "color": '#671800'
    },
    8: {
        "name": "Phacoemulsification Tip",
        "color": '#009B55'
    },
    9: {
        "name": "Irrigation-Aspiration",
        "color": '#F7921D'
    },
    10: {
        "name": "Lens Injector",
        "color": '#613F99'
    },
    11: {
        "name": "Incision Knife",
        "color": '#46C5DD'
    },
    12: {
        "name": "Katena Forceps",
        "color": '#EE2967'
    },
    13: {
        "name": "Capsulorhexis Forceps",
        "color": '#0071BC'
    }
}

def get_cataract1k_colormap():
    colors = []
    for i in range(len(CLASSES)):
        rgb_color = ImageColor.getrgb(CLASSES[i]['color'])
        colors.append(rgb_color)
    return np.array(colors)

def get_cataract1k_float_cmap():
    return torch.from_numpy(get_cataract1k_colormap())/255.0


def get_cadis_colormap():
    """
    Returns cadis colormap as in paper
    :return: ndarray of rgb colors
    """
    return np.asarray(
        [
            [0, 137, 255],
            [255, 165, 0],
            [255, 156, 201],
            [99, 0, 255],
            [255, 0, 0],
            [255, 0, 165],
            [255, 255, 255],
            [141, 141, 141],
            [255, 218, 0],
            [173, 156, 255],
            [73, 73, 73],
            [250, 213, 255],
            [255, 156, 156],
            [99, 255, 0],
            [157, 225, 255],
            [255, 89, 124],
            [173, 255, 156],
            [255, 60, 0],
            [40, 0, 255],
            [170, 124, 0],
            [188, 255, 0],
            [0, 207, 255],
            [0, 255, 207],
            [188, 0, 255],
            [243, 0, 255],
            [0, 203, 108],
            [252, 255, 0],
            [93, 182, 177],
            [0, 81, 203],
            [211, 183, 120],
            [231, 203, 0],
            [0, 124, 255],
            [10, 91, 44],
            [2, 0, 60],
            [0, 144, 2],
            [133, 59, 59],
        ]
    )

def get_cadis_float_cmap():
    return torch.from_numpy(get_cadis_colormap())/255.0


def get_cholecseg8k_colormap():
    """
    Returns cadis colormap as in paper
    :return: ndarray of rgb colors
    """
    return np.asarray(
        [
            [127, 127, 127],  # Black Background
            [210, 140, 140],  # Abdominal Wall
            [255, 114, 114],  # Liver
            [231, 70, 156],  # Gastrointestinal Tract
            [186, 183, 75],  # Fat
            [170, 255, 0],  # Grasper
            [255, 85, 0],  # Connective Tissue
            [255, 0, 0],  # Blood
            [255, 255, 0],  # Cystic Dust
            [169, 255, 184],  # L-hook Electrocautery
            [255, 160, 165],  # Gallblader
            [0, 50, 128],  # Hepatic Vein
            [111, 74, 0],  # Liver Ligament
        ]
    )

def get_cholecseg8k_float_cmap():
    return torch.from_numpy(get_cholecseg8k_colormap())/255.0

