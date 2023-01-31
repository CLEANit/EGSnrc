from ProcessInput import ProcessInput
from pandas import read_csv

#npad = 4
#f = npad + 1

def ProcessInputCSV(csv_path):
    do = read_csv(csv_path, header=None)
    do = do.to_numpy()
    
    # Define shape tuple
    do_shape = [1] + list(do.shape)
    do_shape = tuple(do_shape)
    do = do.reshape(do_shape)
    return do


def DesignToRandomNumber(design, base=4, npad=2):
    ''' Converts designs back into random number. \
            Assumes padding'''
    int_in_base = "".join(("".join(map(str,row_i)) for row_i in design[npad:-npad, npad:-npad]))

    rand_num = int(int_in_base, base)
    return rand_num



def FullProcessSingle(csv_path, npad, mod=""):
    f = npad + 1
    dist_obj = ProcessInputCSV(csv_path) 
    single_windows, si = ProcessInput(dist_obj, npad)
    single_windows = single_windows.reshape((-1,f*f))

    if mod == "":
        return single_windows

    else:
        single_output = mod.predict(single_windows)
        return single_windows, single_output


def PlotSinglePair(csv_path, mod=""):
    import matplotlib.pyplot as plt
    dist_obj = ProcessInputCSV(csv_path) 
    single_windows, si = ProcessInput(dist_obj, 2)
    single_windows = single_windows.reshape((-1,3*3))

    if mod == "":
        try:
            plt.figure()
            plt.imshow(si[0])
            plt.title("Input Object")
            plt.show()
        except TypeError:
            return si

    else:
        single_output = mod.predict(single_windows)
        single_output_image = single_output.reshape((-1,xdim,xdim))
        
        CompareImages(si[0], single_output_image[0], A_title="Input Object", B_title="Prediction")
        #fig, axes = plt.subplot_mosaic("AB")
        #axes['A'].imshow(single_output_image[0])
        #axes['A'].set_title("Prediction")

        #axes['B'].imshow(si[0])
        #axes['B'].set_title("Input Object")
        plt.show()


def MakeCommonColorbar(A, B):
    import numpy as np
    import matplotlib as mpl

    # Calculate range of values
    range_of_values = np.dstack((A, B))
    range_of_values = [np.min(range_of_values), np.max(range_of_values)]

    # Define colors over range
    norm = mpl.colors.Normalize(vmin=range_of_values[0], vmax=range_of_values[1])
    cmap = mpl.cm.inferno
    #cmap = mpl.cm.gist_gray

    cb_mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    #cb_axis = GetSharedCBPos(A,B)

    return norm, cmap, cb_mappable

def GetSharedCBPos(images):
    import numpy as np
    import matplotlib.pyplot as plt
    # Calculate colorbar y-values
    cb_height_param = 0.125
    image_bottom = np.min([np.array(image.get_position())[:,1] for image in iter(images)])
    cb_middle = image_bottom * 3/4#/ 2
    cb_height = image_bottom * cb_height_param
    cb_bottom = cb_middle - cb_height / 2

    # Calculate colorbar x-values
    cb_left = [np.array(image.get_position())[:,0] for image in iter(images)]
    cb_width_full = np.max(cb_left) - np.min(cb_left)

    # leave some non-overlapping room
    empty_fraction = 1/4
    cb_width = cb_width_full * (1 - empty_fraction)


    # Align colour bar so start after half of the empty space.
    cb_left = np.min(cb_left)
    cb_left += cb_width_full * empty_fraction / 2 

    # Make position for colorbar
    cb_axis = plt.axes([cb_left, cb_bottom, cb_width, cb_height])
    return cb_axis



def CompareImages(A, B, A_title="Prediction", B_title="Mean"):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from mpl_toolkits.axes_grid1.mpl_axes import Axes
    import numpy as np

    # Get req's for Common ColorBar
    norm, cmap, cb_mappable = MakeCommonColorbar(A, B)

    # Make figure
    fig, axes = plt.subplot_mosaic('AB')

    # plot each image
    ai = axes["A"].imshow(A, norm=norm, cmap=cmap)
    axes["A"].set_title(A_title)

    bi = axes["B"].imshow(B, norm=norm, cmap=cmap)
    axes["B"].set_title(B_title)


    cb_axis = GetSharedCBPos((axes["A"], axes["B"]))
    # Make colorbar
    #plt.colorbar(ai, cax=cb_axis, orientation='horizontal')
    plt.colorbar(cb_mappable, cax=cb_axis, orientation='horizontal')

    plt.show()
    return fig
