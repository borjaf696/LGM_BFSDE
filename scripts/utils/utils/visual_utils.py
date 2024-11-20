import matplotlib.pyplot as plt
import seaborn as sns

class VisualizationHelper:
    @staticmethod
    def plot_time_series(df, x, value_column, title="Results", xlabel="Xt", ylabel="Y"):
        """
        Grafica una serie de tiempo usando seaborn y matplotlib.

        :param df: DataFrame de Pandas que contiene la serie de tiempo.
        :param date_column: Nombre de la columna en df que contiene las fechas.
        :param value_column: Nombre de la columna en df que contiene los valores.
        :param title: Título del gráfico.
        :param xlabel: Etiqueta del eje X.
        :param ylabel: Etiqueta del eje Y.
        """
        sns.set_theme(style="darkgrid")

        plt.figure(figsize=(6, 6))
        sns.lineplot(data=df, x=x, y=value_column)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Mostrar el gráfico
        plt.show()

    @staticmethod
    def plot_multiple_series(
        df, x, values_column, title="Results", xlabel="Xt", ylabel="Y"
    ):
        """
        Grafica una serie de tiempo usando seaborn y matplotlib.

        :param df: DataFrame de Pandas que contiene la serie de tiempo.
        :param date_column: Nombre de la columna en df que contiene las fechas.
        :param value_column: Nombre de la columna en df que contiene los valores.
        :param title: Título del gráfico.
        :param xlabel: Etiqueta del eje X.
        :param ylabel: Etiqueta del eje Y.
        """
        sns.set(style="darkgrid")

        plt.figure(figsize=(10, 4))
        for y in values_column:
            sns.lineplot(data=df, x=x, y=y, label=y)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()

        # Mostrar el gráfico
        plt.show()
    
    @staticmethod
    def plot_serie(
        df, x, y, title="Results", xlabel="Xt", ylabel="Y", hue = None
    ):
        """
        Grafica una serie de tiempo usando seaborn y matplotlib.

        :param df: DataFrame de Pandas que contiene la serie de tiempo.
        :param date_column: Nombre de la columna en df que contiene las fechas.
        :param value_column: Nombre de la columna en df que contiene los valores.
        :param title: Título del gráfico.
        :param xlabel: Etiqueta del eje X.
        :param ylabel: Etiqueta del eje Y.
        """
        sns.set(style="darkgrid")

        plt.figure(figsize=(10, 4))
        if hue is None:
            sns.lineplot(data=df, x=x, y=y, label=y)
        else:
            sns.lineplot(data=df, x=x, y=y, hue = hue)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()

        # Mostrar el gráfico
        plt.show()