import matplotlib.pyplot as plt
import pickle
import sys

def OpenFigure(filePath):
    fig = pickle.load(open(filePath, 'rb'))
    ax_master = fig.axes[0]
    for ax in fig.axes:
        if ax is not ax_master:
            ax_master.get_shared_y_axes().join(ax_master, ax)
    return fig

def SaveFigure(fig, filePath):
    pickle.dump(fig, open(filePath, 'wb'))
    print("Saved figure to: " + filePath)

if __name__ == "__main__":
    fig = OpenFigure(sys.argv[1])

    # Put plot modifications here!

    plt.show()
