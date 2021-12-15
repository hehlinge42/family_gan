import json
import argparse
import os
import cv2

from logzero import logger

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Just a Fibonacci demonstration")
    parser.add_argument(
        "-r",
        "--root",
        dest="root",
        help="root path of the data",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        help="output filepath",
    )
    args = parser.parse_args()

    images_fp = os.listdir(args.root)
    for image_fp in images_fp:
        logger.debug(f"image_fp: {image_fp}")
        img = cv2.imread(os.path.join(args.root, image_fp), cv2.IMREAD_UNCHANGED)
        resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(args.output, image_fp), resized)
