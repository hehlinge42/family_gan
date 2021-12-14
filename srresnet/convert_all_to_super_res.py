import os
from logzero import logger

from srresnet import recognize_from_image

if __name__ == "__main__":

    ROOT = os.path.join("..", "data")
    FMD_ORIGINAL_PATH = os.path.join(ROOT, "original", "FMD")
    FMS_ORIGINAL_PATH = os.path.join(ROOT, "original", "FMS")
    FMSD_ORIGINAL_PATH = os.path.join(ROOT, "original", "FMSD")

    FMD_SUPERRES_PATH = os.path.join(ROOT, "super_res", "FMD")
    FMS_SUPERRES_PATH = os.path.join(ROOT, "super_res", "FMS")
    FMSD_SUPERRES_PATH = os.path.join(ROOT, "super_res", "FMSD")

    ALL_ORIGINAL_PATHS = {
        FMD_ORIGINAL_PATH: FMD_SUPERRES_PATH,
        FMS_ORIGINAL_PATH: FMS_SUPERRES_PATH,
        FMSD_ORIGINAL_PATH: FMSD_SUPERRES_PATH,
    }

    for original_path, superres_path in ALL_ORIGINAL_PATHS.items():
        logger.debug(f"original_path: {original_path}, superres_path: {superres_path}")
        images_paths = os.listdir(original_path)
        input_filepaths = [os.path.join(original_path, image) for image in images_paths]
        output_filepaths = [
            os.path.join(superres_path, image) for image in images_paths
        ]
        logger.debug(
            f"images_paths: {images_paths}\ninput_filepaths: {input_filepaths}\noutput_filepaths: {output_filepaths}"
        )
        recognize_from_image(input_filepaths, output_filepaths)
