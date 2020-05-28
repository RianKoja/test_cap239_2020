import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genextreme, norm, lognorm
import seaborn as sns

# Example usage:
def plot_ks_gev_gauss(data_sample, alg_name):
    data_min = min(data_sample)
    data_max = max(data_sample)
    n_points = 100
    plot_points = [(data_min + (i/n_points) * (data_max-data_min)) for i in range(0, n_points+1)]

    # Estimate gaussian:
    nrm_fit = norm.fit(data_sample)
    # GEV parameters from fit:
    (mu, sigma) = nrm_fit
    rv_nrm = norm(loc=mu, scale=sigma)
    # Create data from estimated GEV to plot:
    nrm_pdf = rv_nrm.pdf(plot_points)

    # Estimate GEV:
    gev_fit = genextreme.fit(data_sample)
    # GEV parameters from fit:
    c, loc, scale = gev_fit
    rv_gev = genextreme(c, loc=loc, scale=scale)
    # Create data from estimated GEV to plot:
    gev_pdf = rv_gev.pdf(plot_points)

    # Estimate Lognormal:
    lgn_fit = lognorm.fit(data_sample)
    # GEV parameters from fit:
    s, lol, scall = lgn_fit
    rv_lgn = lognorm(c, loc=lol, scale=scall)
    # Create data from estimated GEV to plot:
    lng_pdf = rv_lgn.pdf(plot_points)

    # Make a Kernel density plot and add other fits:
    sns.set(color_codes=True)
    plt.figure()
    ax = sns.kdeplot(data_sample, label='Kernel Density')
    # ax.plot(plot_points, gev_pdf, label='Estimated GEV') # looks awful.
    ax.plot(plot_points, nrm_pdf, label='Estimated Gaussian')
    #ax.plot(plot_points, lng_pdf, label='Estimated Lognormal')
    ax.legend()

    # Use title to indicate parameters found:
    plot_title = "PDF estimated from data created with " + alg_name + "\n"
    #plot_title += "Estimated parameters for GEV: location={:.2f} scale={:.2f} c={:.2f}\n".format(loc, scale, c)
    plot_title += "Estimated parameters for Gaussian: location={:.2f} scale={:.2f}\n".format(mu, sigma)
    #plot_title += "Estimated parameters for Lognorm: shape={:.2f} location={:.2f} scale={:.2f}\n".format(s, lol, scall)

    plt.title(plot_title)
    plt.xlabel("Independent Variable")
    plt.ylabel("Probability Density from " + str(len(data_sample)) + " points")
    plt.tight_layout()
    plt.draw()


if __name__ == '__main__':
    mean, cov = [0, 2], [(1, .5), (.5, 1)]
    series, y = np.random.multivariate_normal(mean, cov, size=100).T
    plot_ks_gev_gauss(series, 'example np.random')

    plt.show()
