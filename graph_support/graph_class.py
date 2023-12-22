from dataclasses import dataclass

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

"""
グラフの生成・保存を行うClass
"""


@dataclass
class Graph:
    @property
    def custom_color(self):
        return self._custom_color

    @custom_color.setter
    def custom_color(self, custom_color: list):
        self._custom_color = custom_color
        if len(self._custom_color) > 0:
            custom_template = go.layout.Template()
            custom_template.layout.colorway = self._custom_color
            pio.templates.default = custom_template

    def create_fig(self):
        self._fig = go.Figure()

    def create_multi_fig(self, row: int, col: int, horizon: float = None):
        self._fig = make_subplots(rows=row, cols=col, horizontal_spacing=horizon)

    def save_fig(self, fig_name: str, url=""):
        self._fig.write_image(f"{url}/{fig_name}.png")

    def update_fig(self, fig):
        self._fig = fig
        return self._fig


"""
グラフをカスタマイズするClass
"""


class PlotlyText(Graph):
    def __init__(self):
        super().__init__()
        self._title: str = ""
        self._barmode: str = "stack"
        self._bargap: float = 0.05
        self._bargroupgap: float = 0.1
        self._plot_bgcolor: str = "white"
        self._paper_bgcolor: str = "white"
        self.width = 600
        self.height = 400
        self._font_color = "black"

    def set_barmode(self, barmode):
        self._barmode = barmode

    def set_bargap(self, bargap):
        self._bargap = bargap

    def set_plot_bgcolor(self, plot_bgcolor):
        self._plot_bgcolor = plot_bgcolor

    def set_paper_bgcolor(self, paper_bgcolor):
        self._paper_bgcolor = paper_bgcolor

    def set_title(self, title: str) -> None:
        self._title = title
        self._fig.update_layout(
            title=self._title,
            title_font=dict(family="Times New Roman", size=20, color="black"),
            title_x=0.5,  # タイトルを中央に揃える
        )

    def set_font_color(self, font_color):
        self._font_color = font_color

    def apply_main_setting(self):
        self._fig.update_layout(
            barmode=self._barmode,
            bargap=self._bargap,
            bargroupgap=self._bargroupgap,
            plot_bgcolor=self._plot_bgcolor,
            paper_bgcolor=self._paper_bgcolor,
            legend=dict(x=1, xanchor="auto", y=1, yanchor="auto"),
            margin=dict(l=60, r=70, t=40, b=0),
            width=self.width,
            height=self.height,
            showlegend=True,
            font=dict(
                family="Courier New, monospace",
                size=15,
                color=self._font_color,
            ),
        )

    def apply_y2_setting(self, y2axis):
        self._fig.update_layout(yaxis2=y2axis)

    def apply_y_setting(self, yaxis):
        self._fig.update_layout(yaxis=yaxis)

    def apply_x_setting(self, xaxis):
        self._fig.update_layout(xaxis=xaxis)

    @property
    def title(self):
        return self._title
