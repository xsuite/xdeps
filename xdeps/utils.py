def mpl_display_png(png):
    import io
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    sio = io.BytesIO()
    sio.write(png)
    sio.seek(0)
    img = mpimg.imread(sio)
    imgplot = plt.imshow(img, aspect='equal')
    plt.xticks([])
    plt.yticks([])

def os_display_png(png):
    import os
    open("/tmp/out.png",'wb').write(png)
    os.system("(display /tmp/out.png;rm /tmp/out.png)&")

def ipy_display_png(png):
    from IPython.display import Image, display
    plt = Image(png)
    display(plt)


