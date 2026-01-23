""" update models in this repository.
    This script finds all osim files in the repo, instantiates an opensim.Model from each
    then call initSystem on it, then overwrites the osim files.
    It is used for the side-effect of writing the osim files in the format/version matching
    the opensim+python version it's run in. For example to upgrade to 4.x format, run in environment of python+opensim4.x

"""
import os
import sys
from fnmatch import fnmatch

import opensim

#import opensim as opensim
root = os.getcwd()
pattern = "*.osim"

osimpaths = []
modelnames = []

for path, _subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            osimpaths.append(os.path.join(path, name))
            modelnames.append(name)

for i in range(len(osimpaths)):
    # Without this next line, the print above does not necessarily
    # precede OpenSim's output.
    sys.stdout.flush()
    filename = osimpaths[i]
    modelname = modelnames[i]
    try:
        model = opensim.Model(filename)
        s = model.initSystem()
    except Exception:
        sys.exit(1)

    # Print the latest model to file
    model.printToXML(filename)
    # Try and read back in the file
    try:
        reloadedModel = opensim.Model(filename)
        s2 = reloadedModel.initSystem()
    except  Exception:
        sys.exit(1)


