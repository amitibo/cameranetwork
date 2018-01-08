from __future__ import division
import argparse
import dateparser
import datetime
import re

IN_PATHS = ["cameralog_170211_094622_camera.txt"]
OUT_PATH = "2017_02_11.csv"

IN_PATHS = ["cameralog_170220_075530_camera.txt"]
OUT_PATH = "2017_02_20.csv"

IN_PATHS = ["cameralog_170810_103932_camera.txt"]
OUT_PATH = "2017_08_10.csv"

def main():

    p = re.compile('\[\s*(\d+\.\d*)\s+(\d+\.\d*)\s*\]')

    output = [",object,pos_x,pos_y,sunshader_angle"]
    for in_path in IN_PATHS:
        with open(in_path, "rb") as f:
            for line in f:
                if not "Centroid" in line:
                    continue
                parts = line.strip().split(",")
                t = dateparser.parse(parts[0]) - datetime.timedelta(seconds=7200)
                x, y = [float(i) for i in p.search(parts[1]).groups()]
                output.append("{},Sun,{},{},1.0".format(t, x, y))

    with open(OUT_PATH, "wb") as f:
        f.write("\n".join(output))


if __name__ == "__main__":
    main()