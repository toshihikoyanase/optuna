from optuna.logging import get_logger
from optuna.study import Study
from optuna.visualization._plotly_imports import _imports

if _imports.is_successful():
    from optuna.visualization._plotly_imports import px, go

_logger = get_logger(__name__)


def plot_timeline(
    study: Study,
) -> "go.Figure":
    _imports.check()
    dataframe = study.trials_dataframe()
    layout = go.Layout(
        title="Timeline Plot",
        xaxis={"title": "Datetime"},
        yaxis={"title": "Trial"},
    )
    fig = px.timeline(
        dataframe,
        x_start="datetime_start",
        x_end="datetime_complete",
        y="number",
    )
    fig.update_layout(layout)
    return fig
