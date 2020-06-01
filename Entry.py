########################################################################################################################
# Prova Rian Koja
########################################################################################################################


# Standard imports:
import matplotlib.pyplot as plt
import pandas as pd

# Local imports:
from tools import getdata, cullen_frey, fit_distribution, specplus, mfdfa_ss, print_table, imc_sf


# Adquirir os dados do país (EUA):
dataframe = getdata.acquire_data()
series = dataframe['new_cases'].tolist()

# Plot histogram:
dataframe['new_cases'].plot.hist(by='new_cases', bins=10, grid=True)
plt.title('Distribution of daily new COVID-19 cases in the USA')
plt.draw()

# Identificar no espaço de  Cullen-Frey:
cullen_frey.series2chart(series)

# Ajustar uma PDF:
fit_distribution.plot_ks_gev_gauss(series, "Daily New USA COVID-19 Cases")

# Calcular o índice espectral:
alpha, beta_theoretical = specplus.main(series)
print("alpha was computed as ", alpha, "and the theoretical beta associated is ", beta_theoretical)

# Obter o espectro de singularidade:
mfdfa_dict = mfdfa_ss.main(series)


# Criar uma tabela:
df = pd.DataFrame()
df['Parameter'] = [r'$\alpha$', r'$\beta$', r'$\Delta\alpha$', r'$\alpha_0$', r'$A_{\alpha}$']
df['Value'] = [alpha, beta_theoretical, mfdfa_dict['delta_alpha'], mfdfa_dict['alpha_zero'], mfdfa_dict['a_alpha']]
print_table.render_mpl_table(df, header_columns=0, col_width=3.0)

# Fazer a parte relacionada ao mode IMC-SF, para dois conjuntos diferentes de p,
# mostrando que a data final de propagação é livre
imc_sf.main(p=[0.5, 0.45, 0.05], date_end='2020-06-18')
imc_sf.main(p=[0.7, 0.25, 0.05])

plt.show()
print("Finished", __file__)
