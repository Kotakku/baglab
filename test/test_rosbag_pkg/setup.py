from setuptools import setup

package_name = "test_rosbag"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/record.launch.py"]),
    ],
    install_requires=["setuptools"],
    entry_points={
        "console_scripts": [
            "test_publisher = test_rosbag.test_publisher:main",
        ],
    },
)
