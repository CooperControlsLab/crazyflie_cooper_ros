[![Build Status](https://travis-ci.org/whoenig/crazyflie_ros.svg?branch=master)](https://travis-ci.org/whoenig/crazyflie_ros)

crazyflie_cooper_ros
=============

ROS stack for Bitcraze Crazyflie (http://www.bitcraze.se/), with the following features:

* Support for Crazyflie 1.0 and Crazyflie 2.0 (using stock firmware)
* Publishes on-board sensors in ROS standard message formats
* Supports ROS parameters to reconfigure crazyflie parameters
* Support for using multiple Crazyflies with a single Crazyradio
* Includes external controller for waypoint navigation (if motion capture system is available)
* No dependency to the Bitcraze SDK (Driver and Controller written in C++)

A tutorial (for a slightly older version) is available in W. Hönig and N. Ayanian. "Flying Multiple UAVs Using ROS", Chapter in Robot Operating System (ROS): The Complete Reference (Volume 2), Springer, 2017. (see http://act.usc.edu/publications.html for a free pre-print).

If you want to control many Crazyflies or look for a good controller for a single Crazyflie, take a look at http://crazyswarm.readthedocs.io/en/latest/. We are currently in the process to unify the Crazyswarm and crazyflie_ros as well as contributing the Crazyswarm firmware changes back to the official firmware.

## Installation (Cooper Specific)

Install VS Code:
`sudo snap install --classic code`

Install ROS, we are unsing ROS Kinetic with Ubuntu Xenial 16.04

Read up through section 3.2 of the following [USC paper](http://act.usc.edu/publications/Hoenig_Springer_ROS2017.pdf). 3.3 details updating the CF firmware, do so if necessary. 

Create a catkin workspace accordingly
```
mkdir -p ~/crazyflie_ws/src
cd ~/crazyflie_ws/src
catkin_init_workspace
```

Clone this repository and update submodules
```
git clone https://github.com/CooperControlsLab/crazyflie_cooper_ros.git
cd crazyflie_cooper_ros
git submodule init
git submodule update
```

Clone vicon_bridge package into workspace
```
cd ~/crazyflie_ws/src
git clone https://github.com/ethz-asl/vicon_bridge.git
```

Use `catkin_make` on your workspace to compile.

Navigate to `~/crazyflie_ws/src/vicon_bridge/launch` and set hostport address to `199.98.17.181:801` in the `vicon.launch` file

See [here](https://wiki.bitcraze.io/doc:crazyflie:dev:fimware:sensor_to_control) for a block diagram of the low-level crazyflie controller

See [here](https://arxiv.org/pdf/1608.05786.pdf) for detailed crazyflie system modeling

## Usage

There are six packages included: crazyflie_cpp, crazyflie_driver, crazyflie_tools, crazyflie_description, crazyflie_controller, and crazyflie_demo.
Note that the below description might be slightly out-of-date, as we continue merging the Crazyswarm and crazyflie_ros.

### Crazyflie_demo (Cooper Demos are here!)

This package contains a rich set of examples to get quickly started with the Crazyflie. The general structure is as follows:

**1 - Launch**: Launches the controller node which publishes `cmd_vel` messages to the crazyflie through use of various off-board controllers. A unique Crazyflie ROS namespace is generated depending on the number of Drones that are desired for the particular demo. 

**2 - Crazyflie Script**: A unique Python file is called by the Launch file, one for each Crazyflie. This file generates a trajectory and instantiates the `CooperativeQuad` class which instructs the high level maneuvers that are desired for the particular Crazyflie flight.

**3a - Trajectories** The Crazflie script calls a trajectory generation algorithm located in `a_traj_generator.py`

**4a - Trajectory Types** The type of trajectory is determined by the various classes contained in `a_traj_generator.py`

**3b - Maneuvers**: The Crazyflie script calls maneuvers which are contained in `a_cooperative_quad.py` which contains the `CooperativeQuad` class. This class holds parameters relevant to each individual Crazyflie and instructions for flight maneuvers such as hovering, trajectory  tracking, and landing.

**4b - Controllers** The `CooperativeQuad` class calls controller classes located in `a_cf_controller_phys.py`. These classes produce the controller commands that are used in the flight maneuvers.

The following image displays which maneuvers and control algorithms are currently available:
![](/crazyflie_demo/scripts/plots/tree.png)

#### Single Drone Hover

**Description**: Runs a hover controller for a single crazyflie at position (0,0,0.3). Uses VICON position feedback for localization. 

**Launch**: Launch the following file with `crazyflie3` enabled in Vicon Tracker.
```
roslaunch crazyflie_demo a_hover_stiff.launch
```
**Notes**: Follows the hover controller described in section 3.1.2 of the following thesis: https://arxiv.org/pdf/1608.05786.pdf

#### Multi Drone Hover

**Description**: Runs the hover controller on three drones

**Launch**: Launch the following file with `crazyflie3`, `crazyflie4`, and `crazyflie5` enabled in Vicon Tracker.
```
roslaunch crazyflie_demo a_hover_stiff_multi.launch
```
**Notes**: Follows the hover controller described in section 3.1.2 of the following thesis: https://arxiv.org/pdf/1608.05786.pdf

#### Single Drone Trajectory Tracking

**Description**: Runs a trajectory tracking algorithm on a single drone

**Launch**: Launch the following file with `crazyflie4` enabled in Vicon Tracker
```
roslaunch crazyflie_demo a_traj_tracking.launch
```

**Notes**: Follows a 2D version of the trajectory tracking controller described in seciton 2.2.2 of the following thesis: https://repository.upenn.edu/cgi/viewcontent.cgi?article=1705&context=edissertations

#### Multi Drone Trajectory Tracking Following Standing Wave Trajectories

**Description**: Runs three drones following a standing wave pattern. Can run `a_traj_generator.py` in `scripts` folder to see wave creation options

**Launch**: Launch the following file with `crazyflie3`, `crazyflie4`, and `crazyflie5` enabled in Vicon Tracker
```
roslaunch crazyflie_demo a_standing_wave_multi.launch
```

**Notes**: Follows the same trajectory tracking algorithm described in demo above

[![](http://img.youtube.com/vi/urXF8xXFFHA/0.jpg)](http://www.youtube.com/watch?v=urXF8xXFFHA "Standing Wave Trajectories")

#### Multi Drone Trajectory Tracking Following Circular Trajectories

[![](http://img.youtube.com/vi/hk8qnfDngmo/0.jpg)](http://www.youtube.com/watch?v=hk8qnfDngmo "Circular Trajectories")

#### Magic Wand Demo

**Description**: Runs an interactive demo where a swarm of Crazyflie drones track a pencil object, see video: https://www.youtube.com/watch?v=J0sCPa3kySQ. 

**Launch**: Launch the following file with `pencil`, `crazyflie3`, `crazyflie4`, and `crazyflie5` objects enabled in Vicon Tracker.
```
roslaunch crazyflie_demo pencil_tracker.launch
```

Start each drone in a seperate terminal window by calling the following services
`rosservice call /crazyflie3/takeoff`
`rosservice call /crazyflie4/takeoff`
`rosservice call /crazyflie5/takeoff`

Can also run `emergency` or `land` services to either cut power or conduct a graceful landing maneuver

**Notes**: Uses the stock cpp controller contained in the Crazyflie_controller directory

[![](http://img.youtube.com/vi/J0sCPa3kySQ/0.jpg)](http://www.youtube.com/watch?v=J0sCPa3kySQ "Magic Wand Demo")

### Crazyflie_demo\model

**Description**: Model of the Crazyflie 2.1 system following dynamics listed in the following paper: https://arxiv.org/pdf/1608.05786.pdf using a Python controller layout described by Randal Beard in the book `Quadrotor Dynamics and Control Rev 0.1`

**Launch**: Run the following file in the `tests` folder
```
python test_controller.py
```

### Crazyflie_Cpp

This package contains a cpp library for the Crazyradio and Crazyflie. It can be used independently of ROS.

### Crazyflie_driver

This package contains the driver. In order to support multiple Crazyflies with a single Crazyradio, there is crazyflie_server (communicating with all Crazyflies) and crazyflie_add to dynamically add Crazyflies.
The server does not communicate to any Crazyflie initially, hence crazyflie_add needs to be used.

### Crazyflie_tools

This package contains tools which are helpful, but not required for normal operation. So far, it just support one tool for scanning for a Crazyflie.

You can find connected Crazyflies using:
```
rosrun crazyflie_tools scan
```

### Crazyflie_description

This package contains a 3D model of the Crazyflie (1.0). This is for visualization purposes in rviz.

### Crazyflie_controller

This package contains a simple PID controller for hovering or waypoint navigation.
It can be used with external motion capture systems, such as VICONhttps://repository.upenn.edu/cgi/viewcontent.cgi?article=1705&context=edissertations.

## ROS Features

### Parameters

The launch file supports the following arguments:
* uri: Specifier for the crazyflie, e.g. radio://0/80/2M
* tf_prefix: tf prefix for the crazyflie frame(s)
* roll_trim: Trim in degrees, e.g. negative if flie drifts to the left
* pitch_trim: Trim in degrees, e.g. negative if flie drifts forward

See http://wiki.bitcraze.se/projects:crazyflie:userguide:tips_and_tricks for details on how to obtain good trim values.

### Subscribers

#### cmd_vel

Similar to the hector_quadrotor, package the fields are used as following:
* linear.y: roll [e.g. -30 to 30 degrees]
* linear.x: pitch [e.g. -30 to 30 degrees]
* angular.z: yawrate [e.g. -200 to 200 degrees/second]
* linear.z: thrust [10000 to 60000 (mapped to PWM output)]

### Publishers

#### imu
* sensor_msgs/IMU
* contains the sensor readings of gyroscope and accelerometer
* The covariance matrices are set to unknown
* orientation is not set (this could be done by the magnetometer readings in the future.)
* update: 10ms (time between crazyflie and ROS not synchronized!)
* can be viewed in rviz

#### temperature
* sensor_msgs/Temperature
* From Barometer (10DOF version only) in degree Celcius (Sensor readings might be higher than expected because the PCB warms up; see http://www.bitcraze.se/2014/02/logging-and-parameter-frameworks-turtorial/)
* update: 100ms (time between crazyflie and ROS not synchronized!)

#### magnetic_field
* sensor_msgs/MagneticField
* update: 100ms (time between crazyflie and ROS not synchronized!)

#### pressure
* Float32
* hPa (or mbar)
* update: 100ms (time between crazyflie and ROS not synchronized!)

#### battery
* Float32
* Volts
* update: 100ms (time between crazyflie and ROS not synchronized!)

## Similar Projects

* https://github.com/gtagency/crazyflie-ros
  * no documentation
  * no teleop
* https://github.com/omwdunkley/crazyflieROS
  * coupled with GUI
  * based on custom firmware
* https://github.com/mbeards/crazyflie-ros
  * incomplete
* https://github.com/utexas-air-fri/crazyflie_fly
  * not based on official SDK
  * no support for logging
* https://github.com/mchenryc/crazyflie
  * no documentation

## Notes

* The dynamic_reconfigure package (http://wiki.ros.org/dynamic_reconfigure/) seems like a good fit to map the parameters, however it has severe limitations:
  * Changed-Callback does not include which parameter(s) were changed. There is only a notion of a level which is a simple bitmask. This would cause that on any parameter change we would need to update all parameters on the Crazyflie.
  * Parameters are statically generated. There are hacks to add parameters at runtime, however those might not work with future versions of dynamic_reconfigure.
  * Groups not fully supported (https://github.com/ros-visualization/rqt_common_plugins/issues/162; This seems to be closed now, however the Indigo binary packages did not pick up the fixes yet).



## Citing This Work

This project is published under the very permissive MIT License. However,
if you use the package we appreciate if you credit this project accordingly.

For academic publications, you can cite the following book chapter:
```
@Inbook{crazyflieROS,
  author={Wolfgang H{\"o}nig
          and Nora Ayanian},
  editor={Anis Koubaa},
  title={Flying Multiple UAVs Using ROS},
  bookTitle={Robot Operating System (ROS): The Complete Reference  (Volume 2)},
  year={2017},
  publisher={Springer International Publishing},
  pages={83--118},
  isbn={978-3-319-54927-9},
  doi={10.1007/978-3-319-54927-9_3},
  url={https://doi.org/10.1007/978-3-319-54927-9_3}
}

```

If your work is related to Mixed Reality, you might cite the paper which introduced the package instead, using the following bibtex entry:
```
@conference{HoenigMixedReality2015,
  author = {Wolfgang H{\"o}nig and Christina Milanes and Lisa Scaria and Thai Phan and Mark Bolas and Nora Ayanian},
  booktitle = {IEEE/RSJ Intl Conf. Intelligent Robots and Systems},
  pages = {5382 - 5387},
  title = {Mixed Reality for Robotics},
  year = {2015}}
```

For any other mentioning please include my affiliation (ACTLab at University of Southern California or USC in short; The link to our webpage is http://act.usc.edu) as this work was partially done as part of my research at USC.
