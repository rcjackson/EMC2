import emc2
import matplotlib.pyplot as plt
import numpy as np

model_path = '../notebooks/allvars.SCM_AWR_linft_BT0_unNa_noaer.nc'
my_model = emc2.core.model.ModelE(model_path)
HSRL = emc2.core.instruments.HSRL()
my_model = emc2.simulator.main.make_simulated_data(my_model, HSRL, 4)

fig, ax = plt.subplots()
Time_2_plot = 17

display = emc2.plotting.SubcolumnDisplay(my_model)
display.plot_subcolumn_timeseries('sub_col_Ze_col_strat')

