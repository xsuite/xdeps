# copyright ############################### #
# This file is part of the Xdeps Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #
from typing import Literal


def mpl_display_image(image_data):
    import io
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    sio = io.BytesIO()
    sio.write(image_data)
    sio.seek(0)
    img = mpimg.imread(sio)
    imgplot = plt.imshow(img, aspect="auto")
    # imgplot = plt.imshow(img)
    plt.xticks([])
    plt.yticks([])


def os_display_image(image_data, ftype="png", remove_file=True):
    import os
    from sys import platform

    fname = f"/tmp/out.{ftype}"
    with open(fname, "wb") as fh:
        fh.write(image_data)

    if remove_file:
        remove_cmd = f"rm {fname}"
    else:
        remove_cmd = "true"

    if platform == "darwin":
        # open doesn't block, wait a little not to remove the file too early
        os.system(f"(open {fname} && sleep 5 && {remove_cmd})&")
    else:
        os.system(f"(display {fname} && {remove_cmd})&")


def ipy_display_image(image_data):
    from IPython.display import Image, display

    plt = Image(image_data)
    display(plt)


def plot_pdot(
        pdot,
        backend: Literal['mpl', 'os', 'ipy'] = 'ipy',
        ftype: Literal['png', 'svg', 'pdf'] = 'png',
        remove_file: bool = True,
):
    if ftype == "png":
        image_data = pdot.create_png()
    elif ftype == "svg":
        image_data = pdot.create_svg()
    elif ftype == "pdf":
        image_data = pdot.create_pdf()
    else:
        raise ValueError(f"Unknown ftype: {ftype}")

    if backend == "mpl" and ftype == "png":
        mpl_display_image(image_data)
    elif backend == "os":
        os_display_image(image_data, ftype=ftype, remove_file=remove_file)
    elif backend == "ipy":
        ipy_display_image(image_data)
    else:
        raise ValueError(f"Unknown backend: {backend} or combination of "
                         f"ftype ({ftype}) and backend invalid.")

    return pdot


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
