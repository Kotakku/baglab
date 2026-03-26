"""Select a rosbag via TUI and plot twist linear x, y, z over time."""

from pathlib import Path

import matplotlib.pyplot as plt

import baglab as bl

bag_path = bl.select_bag(str(Path(__file__).parent / "*"))
print(f"Selected: {bag_path}")

bag = bl.load(bag_path)
twist_df = bag["/test/twist"]

t = bl.stamp_to_sec(twist_df, relative=True)
vel = twist_df.msg.twist.linear.df  # columns: [x, y, z]

fig, axes = plt.subplots(3, 1, sharex=True)
for ax, axis_name in zip(axes, ["x", "y", "z"]):
    ax.plot(t, vel[axis_name])
    ax.set_ylabel(axis_name)
axes[-1].set_xlabel("time [s]")
fig.suptitle(f"twist.linear — {bag_path.name}")
plt.tight_layout()
plt.show()
