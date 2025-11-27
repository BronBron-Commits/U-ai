import glob
import os

def load_all_corpora(root):
    files = glob.glob(os.path.join(root, "core", "*.txt")) + \
            glob.glob(os.path.join(root, "modules", "*.txt"))
    data = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            data.append(fp.read())
    return "\n".join(data)
