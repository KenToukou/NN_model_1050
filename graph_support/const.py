"""
グラフの動的な定数を保存するクラス
"""


class ConstName:
    def __init__(self):
        self._x_name = ""
        self._y_name = ""
        self._y2_name = ""

    @property
    def y_config(self) -> dict:
        return dict(
            title=f"{self._y_name}",
            showline=True,
            showgrid=True,
            gridcolor="grey",  # Set y-axis grid lines to grey
            linecolor="black",
            griddash="dot",  # Set y-axis line to black
        )

    @y_config.setter
    def y_name(self, y_name):
        self._y_name = y_name

    @property
    def x_config(self) -> dict:
        return dict(
            title=f"{self._x_name}",
            showline=True,
            linecolor="black",
            showgrid=True,
            gridcolor="grey",
            griddash="dot",
        )

    @x_config.setter
    def x_name(self, x_name):
        self._x_name = x_name

    @property
    def y2_config(self) -> dict:
        return dict(
            showline=True,
            showgrid=False,
            title=f"{self._y2_name}",
            overlaying="y",
            side="right",
            linecolor="black",
        )

    @y2_config.setter
    def y2_name(self, y2_name):
        self._y2_name = y2_name
