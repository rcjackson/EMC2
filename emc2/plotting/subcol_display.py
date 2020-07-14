import numpy as np
import matplotlib.pyplot as plt

from act.plotting import Display


class SubcolumnDisplay(Display):
    """
    This class contains modules for displaying the generated subcolumn parameters as quicklook
    plots. It is inherited from `ACT <https://arm-doe.github.io/ACT>`_'s Display object. For more
    information on the Display object and its attributes and parameters, click `here
    <https://arm-doe.github.io/ACT/API/generated/act.plotting.plot.Display.html>`_. In addition to the
    methods in :code:`Display`, :code:`SubcolumnDisplay` has the following attributes and methods:

    Attributes
    ----------
    model: emc2.core.Model
        The model object containing the subcolumn data to plot.

    Examples
    --------
    This example makes a four panel plot of 4 subcolumns of EMC^2 simulated reflectivity::

    $ model_display = emc2.plotting.SubcolumnDisplay(my_model, subplot_shape=(2, 2), figsize=(30, 20))
    $ model_display.plot_subcolumn_timeseries('sub_col_Ze_cl_strat', 1, subplot_index=(0, 0))
    $ model_display.plot_subcolumn_timeseries('sub_col_Ze_cl_strat', 2, subplot_index=(1, 0))
    $ model_display.plot_subcolumn_timeseries('sub_col_Ze_cl_strat', 3, subplot_index=(0, 1))
    $ model_display.plot_subcolumn_timeseries('sub_col_Ze_cl_strat', 4, subplot_index=(1, 1))

    """
    def __init__(self, model, **kwargs):
        """

        Parameters
        ----------
        model: emc2.core.Model
            The model containing the subcolumn data to plot.

        Additional keyword arguments are passed into act.plotting.plot.Display's constructor.
        """
        if 'ds_name' not in kwargs.keys():
            ds_name = model.model_name
        else:
            ds_name = kwargs.pop('ds_name')
        super().__init__(model.ds, ds_name=ds_name, **kwargs)
        self.model = model

    def plot_subcolumn_timeseries(self, variable,
                                  column_no, pressure_coords=True, title=None,
                                  subplot_index=(0, ), **kwargs):
        """
        Plots timeseries of subcolumn parameters for a given variable and subcolumn.

        Parameters
        ----------
        variable: str
            The subcolumn variable to plot.
        column_no: int
            The subcolumn number to plot.
        pressure_coords: bool
            Set to true to plot in pressure coordinates, false to height coordinates.
        title: str or None
            The title of the plot. Set to None to have EMC^2 generate a title for you.
        subplot_index: tuple
            The index of the subplot to make the plot in.

        Additional keyword arguments are passed into matplotlib's matplotlib.pyplot.pcolormesh.

        Returns
        -------
        axes: Matplotlib axes handle
            The matplotlib axes handle of the plot.
        """
        ds_name = [x for x in self._arm.keys()][0]
        my_ds = self._arm[ds_name].sel(subcolumn=column_no)
        x_variable = self.model.time_dim
        if pressure_coords:
            y_variable = self.model.height_dim
        else:
            y_variable = self.model.z_field

        x_label = 'Time [UTC]'
        if "long_name" in my_ds[y_variable].attrs and "units" in my_ds[y_variable].attrs:
            y_label = '%s [%s]' % (my_ds[y_variable].attrs["long_name"],
                                   my_ds[y_variable].attrs["units"])
        else:
            y_label = y_variable

        cbar_label = '%s [%s]' % (my_ds[variable].attrs["long_name"], my_ds[variable].attrs["units"])
        if pressure_coords:
            x = my_ds[x_variable].values
            y = my_ds[y_variable].values
            x, y = np.meshgrid(x, y)
        else:
            x = my_ds[x_variable].values
            y = my_ds[y_variable].values.T
            p = my_ds[self.model.height_dim].values
            x, p = np.meshgrid(x, p)
        mesh = self.axes[subplot_index].pcolormesh(x, y, my_ds[variable].values.T, **kwargs)
        if title is None:
            self.axes[subplot_index].set_title(self.model.model_name + ' ' +
                                               np.datetime_as_string(self.model.ds.time[0].values))
        else:
            self.axes[subplot_index].set_title(title)

        if pressure_coords:
            self.axes[subplot_index].invert_yaxis()
        self.axes[subplot_index].set_xlabel(x_label)
        self.axes[subplot_index].set_ylabel(y_label)
        cbar = plt.colorbar(mesh, ax=self.axes[subplot_index])
        cbar.set_label(cbar_label)
        return self.axes[subplot_index]


    def plot_vertical_frequency_dist(self, variable, time=None, pressure_coords=True, title=None,
                                     subplot_index=(0, ), percentile_boundaries=(5, 95), bins=None,
                                     **kwargs):
        """

        Plot distribution of parameters in the subcolumn with height. The solid line will be the mean,
        the shading between the dashed lines represents the data range within the percentiles.

        Parameters
        ----------
        variable: str
            The variable to take the frequency distribution in the vertical
        pressure_coords: bool
            True to plot in pressure coordinates.
        time: tuple of str or None
            Select time period to plot. If this is a tuple, then a start time and end time specified as a string
            in the format of %Y-%m-%d %H:%M:%S must be specified for the interval in the tuple. For example,
            ('2010-10-01 00:00:00', '2010-10-02 00:00:00'). If any of the %H:%M:%S portion of the date string
            is left out, then the missing portions are assumed to be zero. i.e. '2010-10-01' means 00 UTC on
            1 October 2010.
        title: str or None
            The title of the plot
        subplot_index: tuple
            The index of the subplot to make the plot in.
        percentile_boundaries: tuple
            The percentiles for the outer lines of the shaded plot.
        kwargs

        Returns
        -------
        heights, percentiles: float arrays
            The height bins, percentiles of each quantity used in the plot.
        axis: matplotlib axis handle
            The matplotlib axis handle
        """
        ds_name = [x for x in self._arm.keys()][0]

        if pressure_coords:
            y_variable = self.model.height_dim
        else:
            y_variable = self.model.z_field

        if bins is None:
            bins = np.linspace(self.model.ds[y_variable].values.min(), self.model.ds[y_variable].values.max(), 50)

        my_histogram = np.zeros((len(bins)-1, 3))
        if time is None:
            my_ds = self.model.ds[variable]
        else:
            my_ds = self.model.ds[variable].sel(time=slice(time[0], time[1]))
            if not pressure_coords:
                y_variable = self.model.ds[y_variable].sel(time=slice(time[0], time[1]))

        for i in range(len(bins)-1):
            if pressure_coords:
                height_inds = np.logical_and(
                    self.model.ds[self.model.height_dim].values >= bins[i], self.model.ds[self.model.height_dim].values < bins[i+1])
                my_histogram[i, 0] = np.nanpercentile(my_ds.values[:, :, height_inds], percentile_boundaries[0])
                my_histogram[i, 1] = np.nanpercentile(my_ds.values[:, :, height_inds], 50)
                my_histogram[i, 2] = np.nanpercentile(my_ds.values[:, :, height_inds], percentile_boundaries[1])
            else:
                height_inds = np.logical_and(y_variable.values >= bins[i], y_variable.values < bins[i+1])
                my_histogram[i, 0] = np.nanpercentile(my_ds.values[:, height_inds], percentile_boundaries[0])
                my_histogram[i, 1] = np.nanpercentile(my_ds.values[:, height_inds], 50)
                my_histogram[i, 2] = np.nanpercentile(my_ds.values[:, height_inds], percentile_boundaries[1])

        bin_mids = (bins[:-1] + bins[1:]) / 2.0
        self.axes[subplot_index].fill_betweenx(bin_mids, my_histogram[:, 0], my_histogram[:, 2], color='b', alpha=0.5)
        self.axes[subplot_index].plot(my_histogram[:, 1], bin_mids, color='k', linewidth=2)
        if "long_name" in my_ds.attrs and "units" in my_ds.attrs:
            x_label = '%s [%s]' % (my_ds.attrs["long_name"],
                                   my_ds.attrs["units"])
        else:
            x_label = variable

        if "long_name" in self.model.ds[y_variable].attrs and "units" in self.model.ds[y_variable].attrs:
            y_label = '%s [%s]' % (self.model.ds[y_variable].attrs["long_name"],
                                   self.model.ds[y_variable].attrs["units"])
        else:
            y_label = y_variable
        if pressure_coords:
            self.axes[subplot_index].invert_yaxis()
        self.axes[subplot_index].set_ylabel(y_label)
        self.axes[subplot_index].set_xlabel(x_label)

        if title is None:
            title = ds_name + " " + variable
        self.axes[subplot_index].set_title(title)

        return bins, my_histogram, self.axes[subplot_index]
