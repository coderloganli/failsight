import matplotlib.pyplot as plt

from failsight import eda


def test_eda_helpers_produce_frames_and_figures(con):
    sample = eda.sample_drives(con, n_drives=10)
    assert sample["failed"].sum() > 0

    rates = eda.failure_rates_by_model(con)
    assert set(rates["model"]) == {"SYN-MODEL-A", "SYN-MODEL-B"}

    traj = eda.prefailure_trajectories(con, "smart_5_raw", days_before=20)
    assert traj["days_to_failure"].between(0, 20).all()

    for fig in [
        eda.plot_attribute_distributions(sample, ["smart_5_raw", "smart_194_raw"]),
        eda.plot_failure_rates(rates),
        eda.plot_prefailure_trajectory(traj, "smart_5_raw"),
    ]:
        assert fig.axes
        plt.close(fig)
