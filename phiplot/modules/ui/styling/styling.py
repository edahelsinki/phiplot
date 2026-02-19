from dataclasses import dataclass, field


@dataclass
class Styling:
    neutral_gray: str = "#a7afbe"
    contrasting_blue: str = "#05141d"
    plot_blue: str = "#2678B2"

    dark_background: str = "#2b2b2b"
    light_background: str = "#ffffff"

    top_menu_height: int = 40
    accordion_relative_width: str = "95%"
    side_column_relative_width: str = "17.5%"
    middle_column_relative_width: str = "65%"

    bokeh_dark = "dark_minimal"
    bokeh_light = "light_minimal"

    plot_light: dict = field(default_factory = lambda: dict(
        bgcolor="#f8f9fa"
    ))

    plot_dark: dict = field(default_factory = lambda: dict(
        bgcolor="#1e1e1e"
    ))

    border_style: str = field(init=False)
    side_column_style: dict = field(init=False)

    default_button_style: dict = field(init=False)
    default_spinner_style: dict = field(init=False)

    default_spacer_height: int = 10

    default_plot_colors: dict = field(default_factory = lambda: dict(
        palette = "CET_D9",
        fill = "#8b8b8b",
        control_point = "#FF00FF",
        added_point = "#FF5F1F",
        must_link = "#0FFF50",
        cannot_link = "#FF3131"
    ))

    default_button_style: dict = field(default_factory = lambda: dict(
        button_type="primary",
        height=40,
        sizing_mode="stretch_width"
    ))

    default_spinner_style: dict = field(default_factory = lambda: dict(
        size=40,
        color="primary"
    ))

    def __post_init__(self):
        self.border_style = f"1px solid {self.neutral_gray}"
        self.side_column_style = dict(
            width=self.side_column_relative_width,
            height="100%",
            order=self.border_style,
        )
