"""Plot twist linear x, y, z over time."""

from pathlib import Path

import matplotlib.pyplot as plt

import baglab

bag = baglab.load(Path(__file__).parent / "test_bag")
twist_df = bag["/test/twist"]

t = baglab.stamp_to_sec(twist_df, relative=True)
vel = twist_df.msg.twist.linear.df  # columns: [x, y, z]

fig, axes = plt.subplots(3, 1, sharex=True)
for ax, axis_name in zip(axes, ["x", "y", "z"]):
    ax.plot(t, vel[axis_name])
    ax.set_ylabel(axis_name)
axes[-1].set_xlabel("time [s]")
fig.suptitle("twist.linear")
plt.tight_layout()
plt.savefig(Path(__file__).parent / "plot_twist_xyz.png")
plt.show()
