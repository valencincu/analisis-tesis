import matplotlib.pyplot as plt
import numpy as np


class HeightMap:
    def __init__(self, x, y, height_map, angles, calibration_factor):
        self.values = height_map
        self.angles = angles
        self.shape  = height_map.shape
        self.calibration_factor = calibration_factor

        x_length = self.values.shape[1]
        y_length = self.values.shape[0]
        self.x = x
        self.y = y


    def plot_phases(self, fig=None, axs=None, show=True):
        if fig is None:
            fig, axs = plt.subplots(1, 2)
        if axs is None:
            axs = fig.axes

        for i, angles in enumerate(self.angles):
            im = axs[i].contourf(self.x*1e3, self.y*1e3, angles, 100)
            cbar = fig.colorbar(im, ax=axs[i])
            cbar.set_label("Phase [rad]", rotation=270, labelpad=15)
            axs[i].set_xlabel("$x$ [mm]")
            axs[i].set_ylabel("$y$ [mm]")
            axs[i].set_aspect("equal")
            axs[i].set_title(f"Phase {i + 1}")
        plt.tight_layout()

        if show: plt.show()
        return fig


    def plot_map(self, fig=None, ax=None, show=True):

        if fig is None:
            fig, ax = plt.subplots()
        elif ax is None:
            ax = fig.add_subplot(111)
        
        im = ax.pcolormesh(self.x*1e3, self.y*1e3, self.values*1e3)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Height [mm]", rotation=270, labelpad=15)
        ax.set_xlabel("$x$ [mm]")
        ax.set_ylabel("$y$ [mm]")
        ax.set_aspect("equal")

        if show: plt.show()
        return fig


    def plot_slice(self, x_index=None, y_index=None, show=True):
        if (x_index is None) == (y_index is None):
            raise ValueError("Provide exactly one of x_index or y_index.")

        fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(6, 8))

        if y_index is not None:
            sliced = self.values[y_index, :]
            axis   = self.x[y_index, :]

            image = axs[0].pcolormesh(self.x*1e3, self.y*1e3, self.values*1e3)
            fig.colorbar(image, ax=axs[0], label="Height [mm]")

            axs[0].axhline(0, linestyle="--", linewidth=2, color="white")
            axs[1].plot(axis * 1e3, sliced * 1e3)


        else:
            sliced = self.values[:,x_index]
            axis   = self.y[:,x_index]

            image = axs[0].pcolormesh(self.x*1e3, self.y*1e3, self.values*1e3)
            fig.colorbar(image, ax=axs[0], label="Height [mm]")

            axs[0].axvline(0, linestyle="--", linewidth=2, color="white")
            axs[1].plot(axis * 1e3, sliced * 1e3)

        axs[0].set_xlabel(f"$x$ [mm]")
        axs[0].set_ylabel(f"$y$ [mm]")
        axs[0].set_aspect("equal")

        axs[1].set_ylabel("Height [mm]")
        axs[1].set_xlabel("Position [mm]")

        if show: plt.show()
        return fig
