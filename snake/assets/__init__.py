from os.path import dirname, isabs, join

def _resolvePath(path):
    if isabs(path):
        return path

    dir_ = dirname(__file__)
    return join(dir_, "..", path)
