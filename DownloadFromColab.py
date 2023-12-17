import os
import shutil
from argparse import ArgumentParser
from typing import Dict

argparser = ArgumentParser(prog="DownloadFromColab, just gimme the path")

curdir: str = os.path.abspath(os.path.curdir)
argparser.add_argument('-p', '--path', required=True)
argparser.add_argument('-n', '--name', default="archive")
args: Dict = vars(argparser.parse_args())


def packageDir(path: str):
    if (os.path.isdir(path)):
        root_dir, name = os.path.split(path)
        shutil.make_archive(args["name"], "zip", path)
        print("Enjoy...")
    else:
        print("Not a correct path. Exiting..")
        exit(1)


packageDir(os.path.join(curdir, args["path"]))
