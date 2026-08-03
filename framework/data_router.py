from pathlib import Path
import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def identify_data_type(data):
    """
    Identify whether the input is image or network traffic.
    """

    # Image file
    if isinstance(data, str):

        extension = Path(data).suffix.lower()

        if extension in IMAGE_EXTENSIONS:
            return "image"

    # NumPy array
    if isinstance(data, np.ndarray):

        # Image (H,W,C)
        if data.ndim == 3:
            return "image"

        # Network features
        if data.ndim == 1 or data.ndim == 2:
            return "network"

    return "unknown"
