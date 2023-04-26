import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

with open("input.txt", "r") as f:
    # number number \n
    X = np.array([[float(i) for i in line.split()] for line in f.readlines()])
    X = X.reshape((100, 2))

with open("labels.txt", "r") as f:
    # number \n
    y = np.array([float(line) for line in f.readlines()])
    y = y.reshape((100, 1))


with open("output.txt", "r") as f:
    batch = np.array([float(i) for i in f.read().split()])


def show_separation(save=False, name_to_save=""):
    sns.set(style="white")

    xx, yy = np.mgrid[-1.5:2.5:.01, -1.:1.5:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]

    yhat = np.zeros(grid.shape[0])

    yhat = batch

    probs = yhat.reshape(xx.shape)

    f, ax = plt.subplots(figsize=(16, 10))
    ax.set_title("Decision boundary", fontsize=14)
    contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
                          vmin=0, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])

    ax.scatter(X[:,0], X[:, 1], c=y[:], s=50,
               cmap="RdBu", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)

    ax.set(xlabel="$X_1$", ylabel="$X_2$")

    plt.show()

show_separation(save=True, name_to_save="moons_cpp.png")