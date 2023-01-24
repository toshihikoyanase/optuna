import datetime

from optuna._experimental import experimental_func
from optuna.study import Study
from optuna.visualization.matplotlib._matplotlib_imports import _imports


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import dates
    from optuna.visualization.matplotlib._matplotlib_imports import plt


@experimental_func("2.2.0")
def plot_timeline(
    study: Study,
) -> "Axes":
    _imports.check()

    # Set up the graph style.
    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    _, ax = plt.subplots()
    ax.set_title("Timeline Plot")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Trial")

    ax.barh(
        y=[t.number for t in study.trials],
        width=[
                t.datetime_complete - t.datetime_start if t.datetime_complete is not None else datetime.datetime.now() - t.datetime_start 
                for t in study.trials
            ],
        left=[t.datetime_start for t in study.trials],
        color="tab:blue",
    )
    # ax.xaxis_date()
    ax.xaxis.set_major_formatter(dates.DateFormatter("%H:%M:%S"))
    plt.gcf().autofmt_xdate()
    return ax
