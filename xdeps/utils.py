def mpl_display_png(png):
    import io
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    sio = io.BytesIO()
    sio.write(png)
    sio.seek(0)
    img = mpimg.imread(sio)
    imgplot = plt.imshow(img, aspect='auto')
    #imgplot = plt.imshow(img)
    plt.xticks([])
    plt.yticks([])

def os_display_png(png):
    import os
    with open("/tmp/out.png",'wb') as fh:
        fh.write(png)
    os.system("(display /tmp/out.png && rm /tmp/out.png)&")

def ipy_display_png(png):
    from IPython.display import Image, display
    plt = Image(png)
    display(plt)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


