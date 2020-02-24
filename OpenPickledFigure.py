import matplotlib.pyplot as plt
import pickle
import sys
import os

def OpenFigure(filePath):
    fig = pickle.load(open(filePath, 'rb'))
    ax_master = fig.axes[0]
    for ax in fig.axes:
        if ax is not ax_master:
            ax_master.get_shared_y_axes().join(ax_master, ax)
    return fig

def SaveFigure(fig, filePath):
    filename = os.path.basename(filePath)
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        name = name[:255 - len(ext)]
        filename = name+ext
        folder = os.path.dirname(filePath)
        filePath = os.path.join(folder, filename)
    
    pickle.dump(fig, open(filePath, 'wb'))
    print("Saved figure to: " + filePath)

if __name__ == "__main__":
    fig = OpenFigure(sys.argv[1])

    # Put plot modifications here!

    plt.show()
