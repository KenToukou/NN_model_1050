from functools import wraps

from graph_support import ConstName, PlotlyText

"""
グラフの設定を統合するClass
"""


class GraphInfo:
    def __init__(self):
        self.axis_name = ConstName()
        self._graph = PlotlyText()
        self._graph.create_fig()
        self._graph.apply_main_setting()

    @property
    def fig(self):
        return self._graph._fig

    @property
    def title(self):
        return self._graph.title

    @fig.setter
    def x_name(self, x_name: str) -> None:
        self.axis_name.x_name = x_name
        self._graph.apply_x_setting(xaxis=self.axis_name.x_config)

    @fig.setter
    def y_name(self, y_name: str) -> None:
        self.axis_name.y_name = y_name
        self._graph.apply_y_setting(yaxis=self.axis_name.y_config)

    @fig.setter
    def y2_name(self, y2_name: str) -> None:
        self.axis_name.y2_name = y2_name
        self._graph.apply_y2_setting(y2axis=self.axis_name.y2_config)

    @title.setter
    def title_name(self, title: str) -> None:
        self._graph.set_title(title=title)


class MultiGraphInfo:
    def __init__(self, row: int, col: int, horizon=None):
        self.axis_name = ConstName()
        self._graph = PlotlyText()
        self._graph.create_multi_fig(row=row, col=col, horizon=horizon)
        self._graph.apply_main_setting()

    @property
    def fig(self):
        return self._graph._fig

    @property
    def title(self):
        return self._graph.title

    @title.setter
    def title_name(self, title: str) -> None:
        self._graph.set_title(title=title)


"""
以下はPlotlyのインスタンスを初期化するためのデコレータ
"""


# decorator function
def before_func(title: str = "", x_name: str = "", y_name: str = "", y2_name: str = ""):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            graph_class = GraphInfo()
            graph_class.x_name = x_name
            graph_class.y_name = y_name
            graph_class.y2_name = y2_name
            graph_class.title_name = title

            return func(*args, **kwargs, graph=graph_class)

        return wrapper

    return decorator
