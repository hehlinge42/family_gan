import json
import argparse
import os

from logzero import logger


def get_ids(filepaths):
    splits = [filepath.split("-") for filepath in filepaths]
    ids = [split[1] for split in splits]
    return list(set(ids))


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

    dirs = os.listdir(args.root)
    logger.debug(f"dirs: {dirs}")
    json_dict = {}

    for dir in dirs:
        logger.debug(f"dir: {dir}")
        filepaths = os.listdir(os.path.join(args.root, dir))
        ids = get_ids(filepaths)
        ids = [dir + "-" + str(idx) for idx in ids]

        for idx in ids:
            if dir == "FMD":
                json_dict[idx] = {
                    "father": os.path.join(args.root, dir, idx + "-F.jpg"),
                    "mother": os.path.join(args.root, dir, idx + "-M.jpg"),
                    "daughter": os.path.join(args.root, dir, idx + "-D.jpg"),
                }
            elif dir == "FMS":
                json_dict[idx] = {
                    "father": os.path.join(args.root, dir, idx + "-F.jpg"),
                    "mother": os.path.join(args.root, dir, idx + "-M.jpg"),
                    "son": os.path.join(args.root, dir, idx + "-S.jpg"),
                }
            elif dir == "FMSD":
                json_dict[idx] = {
                    "father": os.path.join(args.root, dir, idx + "-F.jpg"),
                    "mother": os.path.join(args.root, dir, idx + "-M.jpg"),
                    "daughter": os.path.join(args.root, dir, idx + "-D.jpg"),
                    "son": os.path.join(args.root, dir, idx + "-S.jpg"),
                }

    with open(args.output, "w") as f:
        json.dump(json_dict, f)
