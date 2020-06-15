########################################################################################################################
# Prova Rian Koja
########################################################################################################################


# Standard imports:
import matplotlib.pyplot as plt
import pandas as pd

# Local imports:
from tools import getdata, createdocument, cullen_frey, fit_distribution, specplus, mfdfa_ss, print_table, imc_sf

# Add figures to a wordfile, to be created on GitHub:
doc = createdocument.ReportDocument()
doc.add_heading("Test by Rian", level=0)
doc.add_heading("Part A:", level=1)
# Adquirir os dados do país (EUA):
dataframe = getdata.acquire_data()
series = dataframe['new_cases'].tolist()

# Plot histogram:
dataframe['new_cases'].plot.hist(by='new_cases', bins=10, grid=True)
plt.title('Distribution of daily new COVID-19 cases in the USA')
plt.draw()
doc.add_fig()

# Identificar no espaço de  Cullen-Frey:
cullen_frey.series2chart(series)
doc.add_fig()

# Ajustar uma PDF:
fit_distribution.plot_ks_gev_gauss(series, "Daily New USA COVID-19 Cases")
doc.add_fig()

# Calcular o índice espectral:
alpha, beta_theoretical = specplus.main(series)
print("alpha was computed as ", alpha, "and the theoretical beta associated is ", beta_theoretical)
doc.add_fig()

# Obter o espectro de singularidade:
mfdfa_dict = mfdfa_ss.main(series)
doc.add_fig()


# Criar uma tabela:
df = pd.DataFrame()
df['Parameter'] = [r'$\alpha$', r'$\beta$', r'$\Delta\alpha$', r'$\alpha_0$', r'$A_{\alpha}$']
df['Value'] = [alpha, beta_theoretical, mfdfa_dict['delta_alpha'], mfdfa_dict['alpha_zero'], mfdfa_dict['a_alpha']]
print_table.render_mpl_table(df, header_columns=0, col_width=3.0)
doc.add_paragraph("Summarizing in a table:")
doc.add_fig()

doc.add_heading("Part B:", level=1)
# Fazer a parte relacionada ao modelo IMC-SF, para dois conjuntos diferentes de p,
# mostrando que a data final de propagação é livre
doc.add_paragraph("Using the first suggested p vector:")
imc_sf.main(p=[0.5, 0.45, 0.05], date_end='2020-06-18', doc=doc)

# Use the other proposed p-value:
doc.add_paragraph("Using the second suggested p vector:")
imc_sf.main(p=[0.7, 0.25, 0.05], doc=doc)


# Use a custom p value just to get a cute graph:
doc.add_paragraph("Here, I'll try a different p vector whose entries do not sum 1, just so I can get apparently good" +
                  "predictions:")
imc_sf.main(p=[0.675/2, 0.275/2, 0.05/2], doc=doc)


doc.finish()

plt.show()
print("Finished", __file__)
