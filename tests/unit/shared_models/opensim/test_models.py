"""test models in the distribution
this script finds all osim files in the repo, instantiate opensim.Model from each
then call initSystem on it, roundtrips save/restore to compare.
"""

import logging
import os
import sys
from fnmatch import fnmatch

import opensim

logger = logging.getLogger(__name__)
# import opensim as opensim
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
    logger.info(
        "\n\n\n" + 80 * "=" + f"\nLoading model '{osimpaths[i]}'" + "\n" + 80 * "-"
    )
    # Without this next line, the print above does not necessarily
    # precede OpenSim's output.
    sys.stdout.flush()
    filename = osimpaths[i]
    modelname = modelnames[i]
    try:
        model = opensim.Model(filename)
        s = model.initSystem()
    except (ImportError, OSError):
        logger.info(f"Oops, Model '{modelname}' failed:\n{e.message}")
        sys.exit(1)

    # Print the 4.0 version to file and then read back into memory
    filename_new = filename.replace(".osim", "_new.osim")
    modelname_new = modelname.replace(".osim", "_new.osim")
    # Print the 4.0 model to file
    model.printToXML(filename_new)
    # Try and read back in the file
    try:
        reloadedModel = opensim.Model(filename_new)
        s2 = reloadedModel.initSystem()
    except (ImportError, OSError):
        logger.info(f"Oops, 4.0 written Model '{modelname_new}' failed:\n{e.message}")
        sys.exit(1)

    # Remove the printed file
    os.remove(filename_new)

    if not reloadedModel.isEqualTo(model):
        logger.info(
            f"Initial instance of '{modelname}' is not equal to :\n{modelname_new}"
        )
        raise Exception("Compared two instances of 4.0 models are not equal")

logger.info("All models loaded successfully.")
