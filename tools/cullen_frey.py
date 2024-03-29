########################################################################################################################
# Print Cullen-Frey chart
#
# Based on https://github.com/reinaldo-rosa-inpe/cap239/blob/master/Codigos/cullen_frey_giovanni.py
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.stats import skew, kurtosis


def cullenfrey_chart(xd, yd, legend, title):
    fig, ax = plt.subplots()
    maior = max(xd) * 1.1
    poly_x1 = maior if maior > 4.4 else 4.4
    poly_y1 = poly_x1 + 1
    poly_y2 = 3/2.*poly_x1 + 3
    y_lim = poly_y2 if poly_y2 > 10 else 10
    y_lim = max([y_lim, max(yd) * 1.1])

    x = [0, poly_x1, poly_x1, 0]
    y = [1, poly_y1, poly_y2, 3]

    scale = 1
    step = max([0.1, maior/1000])
    poly = Polygon(np.c_[x, y]*scale, facecolor='#1B9AAA', edgecolor='#1B9AAA', alpha=0.5)
    ax.add_patch(poly)
    ax.plot(xd, yd, marker="o", c="#e86a92", label=legend, linestyle='')
    ax.plot(0, 4.187999875999753, label="logistic", marker='+', c='black', linestyle='None')
    ax.plot(0, 1.7962675925351856, label="uniform", marker='^', c='black', linestyle='None')
    ax.plot(4, 9, label="exponential", marker='s', c='black', linestyle='None')
    ax.plot(0, 3, label="normal", marker='*', c='black', linestyle='None')
    ax.plot(np.arange(0, poly_x1, step), 3/2. * np.arange(0, poly_x1, step) + 3, label="gamma", linestyle='-',
            c='black')
    ax.plot(np.arange(0, poly_x1, step), 2 * np.arange(0, poly_x1, step) + 3, label="lognormal", linestyle='-.',
            c='black')
    ax.legend()
    ax.set_ylim(y_lim, 0)
    ax.set_xlim(-0.2, poly_x1)
    plt.xlabel("Skewness²")
    plt.title(title + ": Cullen and Frey map")
    plt.ylabel("Kurtosis")
    # plt.savefig((title + legend + "cullenfrey.png").replace(" ", "_"))
    plt.draw()


def series2chart(series):
    skewness_square = (skew(series)) ** 2
    kurt = kurtosis(series, fisher=False)
    print("skewness_square =", skewness_square, " and kurtosis = ", kurtosis)
    cullenfrey_chart([skewness_square], [kurt], "Data for USA", "Daily new confirmed COVID-19 cases")


# Example usage:
if __name__ == "__main__":
    cullenfrey_chart([1, 2], [1, 2], "Example legend", "Example title")
    plt.show()
