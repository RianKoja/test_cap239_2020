########################################################################################################################
# Prova Rian Koja
########################################################################################################################


# Standard imports:
import matplotlib.pyplot as plt


# Local imports:
from tools import getdata
from tools import cullen_frey
from tools import exercise_4_2
from tools import specplus

# Adquirir os dados do país (EUA):
dataframe = getdata.acquire_data()
series = dataframe['new_cases'].tolist()

# Plot histogram:
dataframe['new_cases'].plot.hist(by='new_cases', bins=10, grid=True)
plt.draw()

# Identificar no espaço de  Cullen-Frey:
cullen_frey.series2chart(series)

# Ajustar uma PDF:
exercise_4_2.plot_ks_gev_gauss(series, "Daily New USA COVID-19 Cases")

# Calcular o índice espectral:
alpha, beta_theoretical = specplus.main(series)
print("alpha was computed as ", alpha, "and the theoretical beta associated is ", beta_theoretical)

#

plt.show()
print("Finished", __file__)
