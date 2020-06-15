########################################################################################################################
# IMC_SF model for Covid-19 prediction. The Master equation is a method implemented in the RefData class.
#
# Written by Rian Koja to publish in a GitHub repository with specified license.
########################################################################################################################

# Standard imports:
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Local imports:
from tools import getdata


# Use a function to define initial g and s:
def ims_sf_init():
    g0 = 0.8
    s0 = 0
    return g0, s0


class RefData:
    def __init__(self, date_ini=None, date_end=None, p=None):
        if p is None:
            self.p = [0.5, 0.45, 0.05]
        else:
            self.p = p
        if date_ini is None and date_end is None:
            self.df = getdata.acquire_data()  # Using default settings
        elif date_ini is None:
            self.df = getdata.acquire_data(date_end=date_end)
        elif date_end is None:
            self.df = getdata.acquire_data(date_ini=date_ini)
        else:
            self.df = getdata.acquire_data(date_ini=date_ini, date_end=date_end)

        # Typecast dates to datetime, to facilitate plotting:
        self.df['date'] = pd.to_datetime(self.df['date'])
        # Add seven day rolling average:
        self.df['new_cases_7ra'] = self.df['new_cases'].rolling(7, min_periods=1).mean()
        # Initialize new columns:
        g0, s0 = ims_sf_init()
        self.df.at[0, 'g'] = g0
        self.df.at[0, 's'] = s0
        self.df.at[0, 'n_s_min'] = self.df.iloc[0]['new_cases']
        self.df.at[0, 'n_s_max'] = self.df.iloc[0]['new_cases']
        for ii in range(1, len(self.df)):
            g, s, n_s_min, n_s_max = self.ims_sf_master_equation(self.df.iloc[ii-1]['new_cases'],
                                                                 self.df.iloc[ii-1]['new_cases_7ra'],
                                                                 self.df.iloc[ii-1]['g'])
            self.df.at[ii, 'g'] = g
            self.df.at[ii, 's'] = s
            self.df.at[ii, 'n_s_min'] = n_s_min
            self.df.at[ii, 'n_s_max'] = n_s_max

        # Create average column
        self.df['n_s_avg'] = self.df.apply(lambda row: (row.n_s_min+row.n_s_max)/2, axis=1)
        self.df['g_7ra'] = self.df['g'].rolling(7, min_periods=1).mean()
        self.df['s_7ra'] = self.df['s'].rolling(7, min_periods=1).mean()

        # Create a forecast dataframe:
        self.forecast_df = None

    def forecast(self, date_end='2020-06-05'):
        date_last = self.df.iloc[-1]['date']
        days_to_propagate = int((pd.to_datetime(date_end) - date_last) / np.timedelta64(1, 'D'))
        self.forecast_df = pd.DataFrame({'date': pd.date_range(self.df.iloc[-1]['date'] + np.timedelta64(1, 'D'),
                                                               periods=days_to_propagate, freq='D')})
        # Initialize new columns:
        g, s, n_s_min, n_s_max = self.ims_sf_master_equation(self.df.iloc[-1]['new_cases'],
                                                             self.df.iloc[-1]['new_cases_7ra'],
                                                             self.df.iloc[-1]['g'])
        self.forecast_df['g'] = self.df.iloc[-8:-1]['g'].values.mean()
        self.forecast_df['s'] = self.df.iloc[-8:-1]['s'].values.mean()
        self.forecast_df['n_s_min'] = n_s_min
        self.forecast_df['n_s_max'] = n_s_max
        self.forecast_df['new_cases'] = (n_s_max + n_s_min)/2
        # Initialize moving average:
        self.forecast_df['new_cases_7ra'] = 0
        for kk in range(0, 7):
            self.forecast_df.at[kk, 'new_cases_7ra'] = sum(self.df.iloc[-8+kk:-1]['new_cases'].values) / 7.0
        # Perform propagation with self-updating g:
        for ii in range(1, days_to_propagate):
            g, s, n_s_min, n_s_max = self.ims_sf_master_equation(self.forecast_df.iloc[ii-1]['new_cases'],
                                                                 self.forecast_df.iloc[ii-1]['new_cases_7ra'],
                                                                 self.forecast_df.iloc[ii-1]['g'])
            self.forecast_df.at[ii, 'g'] = g
            self.forecast_df.at[ii, 's'] = s
            self.forecast_df.at[ii, 'n_s_min'] = n_s_min
            self.forecast_df.at[ii, 'n_s_max'] = n_s_max
            self.forecast_df.at[ii, 'new_cases'] = (n_s_max + n_s_min)/2
            for nn in range(0, min(7, days_to_propagate-ii)):
                self.forecast_df.at[ii + nn, 'new_cases_7ra'] += self.forecast_df.iloc[ii]['new_cases'] / 7.0

    # Master equation for the model:
    def ims_sf_master_equation(self, n_kt, n_nb_7ra, g0=0.5):
        p1 = self.p[0]
        p2 = self.p[1]
        p3 = self.p[2]

        # Equations 3, 4 and 5:
        n1 = p1 * n_kt
        n2 = p2 * n_kt
        n3 = p3 * n_kt

        if n_kt > n_nb_7ra:
            # Equation 6:
            if n_kt > 0:
                g = n_nb_7ra / n_kt
            else:
                g = 0
        else:
            # Equation 7:
            if n_nb_7ra > 1e-10:
                g = n_kt / n_nb_7ra
            else:
                g = 0

        # Computing derivatives of g:
        if g0 < g:
            q_g = (1 - g) ** 2
            delta_g = (g0 - g) - q_g
        else:
            q_g0 = (1 - g0) ** 2
            delta_g = (g0 - g) + q_g0

        # Equation 1:
        n_s_min = g * (1 * n1 + 3 * n2 + 5 * n3) / 1

        # Equation 2:
        n_s_max = g * (2 * n1 + 4 * n2 + 6 * n3) / 1

        # Equation 8: Use a min to prevent singularity:
        delta_nk = min([(n_nb_7ra - n_kt) / n_kt, n_kt])

        # Equation 9:
        s = (2 * delta_g + delta_nk) / 3.0

        return g, s, n_s_min, n_s_max


def main(date_end='2020-06-13', p=None, doc=None):
    if p is None:
        p = [0.5, 0.45, 0.05]
    # Get data:
    data_test = RefData(p=p)
    data_comp = RefData(date_ini='2020-05-21', date_end=date_end, p=p)
    # Slice beginning of data_comp data so the moving average is contiguous to what it was before:
    data_comp.df = data_comp.df.drop(data_comp.df.index[0:7])

    # Create propagation:
    data_test.forecast(date_end=date_end)

    # Make plot for new cases:
    cols_plot = ['new_cases', 'n_s_avg', 'new_cases_7ra']
    ax = data_test.df.plot(x='date', y=cols_plot, label=[r'$N_{kt}$', r'$N_{s_{avg}}$', r'$<N_{nb}>_{7}$'],
                           color=['g', 'c', 'y'])
    ax.fill_between(data_test.df['date'].values, data_test.df['n_s_min'], data_test.df['n_s_max'], alpha=0.2,
                    label=r'$N_{min}$ $N_{max}$ band')
    # Add comparison data (labelled future)
    data_comp.df.plot(x='date', y=['new_cases', 'new_cases_7ra'], ax=ax,
                      label=[r'$N_{kt}$ ("future")', r'$<N_{nb}>_{7}$ ("future")'])
    # Add forecast data:
    data_test.forecast_df.plot(x='date', y=['new_cases', 'new_cases_7ra'], linestyle='--', color=['c', 'y'],
                               label=[r'$\widehat{N_{kt}}$', r'$\widehat{<N_{nb}>_{7}}$'], ax=ax)
    ax.fill_between(data_test.forecast_df['date'].values, data_test.forecast_df['n_s_min'],
                    data_test.forecast_df['n_s_max'], alpha=0.2, label=r'$N_{min}$ $N_{max}$ forecast band', color='g')
    # Format chart:
    plt.grid('both')
    plt.title("New Cases of COVID-19 in US\nData and Forecasting p = " + str(p))
    plt.ylabel('Number of new cases')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.draw()
    if doc is not None:
        doc.add_fig()

    # Plot g(t) and s(t) curves:
    ax_gs = data_test.df.plot(x='date', y=['g', 's'], label=[r'$g(t)$', r'$s(t)$'], color=['r', 'g'])
    data_test.df.plot(x='date', y=['g_7ra', 's_7ra'], label=[r'$<g(t)>_7$', r'$<s(t)>_7$'],  ax=ax_gs, color=['r', 'g'],
                      linestyle='--')
    data_test.forecast_df.plot(x='date', y=['g', 's'], label=[r'$\widehat{g(t)}$', r'$\widehat{s(t)}$'],
                               color=['purple', 'blue'], ax=ax_gs)
    plt.grid('both')
    plt.title(r'$g(t)$, $s(t)$ and 7 day moving averages. p = ' + str(p))
    plt.legend()
    plt.draw()
    if doc is not None:
        doc.add_fig()


# Sample execution:
if __name__ == "__main__":
    n_kt_test = 30
    n_nb_7ra_test = 45
    g0_test = 0.5

    p_test = [0.5, 0.45, 0.05]
    # p_test = [0.7, 0.25, 0.05]

    print("Run master equation for test, and print result:")
    sample_object = RefData(p=p_test)
    print(sample_object.ims_sf_master_equation(n_kt_test, n_nb_7ra_test, g0=g0_test))

    # Run main function with default arguments:
    main()

    #  Show graphs:
    plt.show()
