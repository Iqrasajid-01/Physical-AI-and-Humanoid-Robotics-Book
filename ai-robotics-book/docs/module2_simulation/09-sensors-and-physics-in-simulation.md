---
title: Sensors and Physics in Simulation
sidebar_label: 09 - Sensors and Physics in Simulation
---

# Sensors and Physics in Simulation

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand how physics engines simulate real-world physics in Gazebo
- Configure and implement various sensor types in simulation
- Model sensor noise and uncertainty in simulated environments
- Integrate physics properties with robot models for realistic simulation
- Evaluate the accuracy and limitations of simulated sensors and physics
- Optimize simulation performance while maintaining accuracy

## Introduction

Realistic simulation of sensors and physics is crucial for developing and testing robotic systems. Gazebo provides sophisticated physics engines and sensor simulation capabilities that allow developers to create environments that closely mimic real-world conditions. Understanding how to configure these elements properly is essential for effective sim-to-real transfer.

The quality of sensor and physics simulation directly impacts the validity of testing results and the success of algorithms when deployed on real robots. This chapter explores the underlying principles and practical implementation of sensors and physics in Gazebo simulation.

## Core Concepts

### Physics Engines in Gazebo

Gazebo supports multiple physics engines:
- **ODE (Open Dynamics Engine)**: Default engine, good for most applications
- **Bullet**: Fast, stable, good for articulated bodies
- **DART (Dynamic Animation and Robotics Toolkit)**: Advanced contact modeling

### Physics Properties

Key physics parameters include:
- **Gravity**: Global gravitational acceleration
- **Friction**: Surface interaction properties (mu1, mu2)
- **Damping**: Energy dissipation parameters
- **Mass and Inertia**: Dynamic properties of objects
- **Collision Properties**: Contact behavior and restitution

### Sensor Types in Gazebo

Gazebo provides simulation for various sensor types:
- **Camera Sensors**: RGB, depth, and stereo cameras
- **LIDAR**: 2D and 3D laser range finders
- **IMU**: Inertial measurement units
- **GPS**: Global positioning system
- **Force/Torque**: Joint force and torque sensors
- **Contact Sensors**: Collision detection sensors

### Sensor Noise Modeling

Realistic sensor simulation includes:
- **Gaussian Noise**: Random measurement errors
- **Bias**: Systematic measurement errors
- **Drift**: Time-varying systematic errors
- **Latency**: Time delays in sensor readings
- **Resolution**: Discretization effects

## Architecture Diagram

![Architecture Diagram](/img/ch9-ad.svg)

<!-- ```mermaid
graph TB
    subgraph "Physics Simulation"
        A[Physics Engine]
        B[Collision Detection]
        C[Contact Processing]
        D[Force Application]
    end

    subgraph "Sensor Simulation"
        E[Camera Sensor]
        F[LIDAR Sensor]
        G[IMU Sensor]
        H[GPS Sensor]
        I[Force/Torque Sensor]
    end

    subgraph "Noise & Uncertainty"
        J[Gaussian Noise]
        K[Bias Models]
        L[Drift Simulation]
        M[Latency Modeling]
    end

    subgraph "Integration"
        N[ROS2 Bridge]
        O[Message Publishing]
        P[Parameter Server]
    end

    A -/-> B
    A -/-> C
    A -/-> D
    B -/-> E
    B -/-> F
    B -/-> G
    B -/-> H
    B -/-> I
    J -/-> E
    K -/-> F
    L -/-> G
    M -/-> H
    J -/-> I
    E -/-> N
    F -/-> N
    G -/-> N
    H -/-> N
    I -/-> N
    N -/-> O
    P -/-> A
    P -/-> E
    P -/-> F
    P -/-> G
    P -/-> H
    P -/-> I
``` -->

## Flow Diagram

![Architecture Diagram](/img/ch9-flow.svg)

<!-- ```mermaid
sequenceDiagram
    participant Phys as Physics Engine
    participant Sensor as Sensor Simulation
    participant Noise as Noise Model
    participant ROS as ROS2 Interface
    participant App as Application

    Phys->>Sensor: Provide physics state
    Sensor->>Noise: Apply noise model
    Noise->>ROS: Generate noisy sensor data
    ROS->>App: Publish sensor messages
    App->>ROS: Send control commands
    ROS->>Phys: Apply forces/torques
    Phys->>Phys: Update physics state
``` -->

## Code Example: Sensor Configuration in URDF

Here's an example of configuring various sensors in a URDF model:

```xml
<?xml version="1.0"?>
<robot name="robot_with_sensors" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.2" length="0.15"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.2" length="0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.4" ixy="0" ixz="0" iyy="0.4" iyz="0" izz="0.2"/>
    </inertial>
  </link>

  <!-- Camera link -->
  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Camera joint -->
  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="0.15 0 0.1" rpy="0 0 0"/>
  </joint>

  <!-- IMU link -->
  <link name="imu_link">
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- IMU joint -->
  <joint name="imu_joint" type="fixed">
    <parent link="base_link"/>
    <child link="imu_link"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
  </joint>

  <!-- LIDAR link -->
  <link name="lidar_link">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.04"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <!-- LIDAR joint -->
  <joint name="lidar_joint" type="fixed">
    <parent link="base_link"/>
    <child link="lidar_link"/>
    <origin xyz="0.15 0 0.1" rpy="0 0 0"/>
  </joint>

  <!-- Gazebo plugins for sensors -->
  <gazebo reference="camera_link">
    <sensor type="camera" name="camera1">
      <update_rate>30.0</update_rate>
      <camera name="head">
        <horizontal_fov>1.3962634</horizontal_fov>
        <image>
          <width>800</width>
          <height>600</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.02</near>
          <far>300</far>
        </clip>
        <noise>
          <type>gaussian</type>
          <mean>0.0</mean>
          <stddev>0.007</stddev>
        </noise>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <ros>
          <namespace>/camera</namespace>
          <remapping>~/image_raw:=image_raw</remapping>
          <remapping>~/camera_info:=camera_info</remapping>
        </ros>
        <camera_name>camera</camera_name>
        <image_topic_name>image_raw</image_topic_name>
        <camera_info_topic_name>camera_info</camera_info_topic_name>
        <frame_name>camera_link</frame_name>
        <hack_baseline>0.07</hack_baseline>
        <distortion_k1>0.0</distortion_k1>
        <distortion_k2>0.0</distortion_k2>
        <distortion_k3>0.0</distortion_k3>
        <distortion_t1>0.0</distortion_t1>
        <distortion_t2>0.0</distortion_t2>
      </plugin>
    </sensor>
  </gazebo>

  <gazebo reference="imu_link">
    <sensor name="imu_sensor" type="imu">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <visualize>true</visualize>
      <imu>
        <angular_velocity>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
              <bias_mean>0.0000075</bias_mean>
              <bias_stddev>0.0000008</bias_stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
              <bias_mean>0.0000075</bias_mean>
              <bias_stddev>0.0000008</bias_stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
              <bias_mean>0.0000075</bias_mean>
              <bias_stddev>0.0000008</bias_stddev>
            </noise>
          </z>
        </angular_velocity>
        <linear_acceleration>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
              <bias_mean>0.1</bias_mean>
              <bias_stddev>0.001</bias_stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
              <bias_mean>0.1</bias_mean>
              <bias_stddev>0.001</bias_stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
              <bias_mean>0.1</bias_mean>
              <bias_stddev>0.001</bias_stddev>
            </noise>
          </z>
        </linear_acceleration>
      </imu>
      <plugin name="imu_plugin" filename="libgazebo_ros_imu_sensor.so">
        <ros>
          <namespace>/imu</namespace>
          <remapping>~/out:=data</remapping>
        </ros>
        <frame_name>imu_link</frame_name>
        <initial_orientation_as_reference>false</initial_orientation_as_reference>
      </plugin>
    </sensor>
  </gazebo>

  <gazebo reference="lidar_link">
    <sensor name="lidar_sensor" type="ray">
      <always_on>true</always_on>
      <visualize>true</visualize>
      <update_rate>10</update_rate>
      <ray>
        <scan>
          <horizontal>
            <samples>720</samples>
            <resolution>1</resolution>
            <min_angle>-1.570796</min_angle>
            <max_angle>1.570796</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.10</min>
          <max>30.0</max>
          <resolution>0.01</resolution>
        </range>
        <noise>
          <type>gaussian</type>
          <mean>0.0</mean>
          <stddev>0.01</stddev>
        </noise>
      </ray>
      <plugin name="lidar_plugin" filename="libgazebo_ros_ray_sensor.so">
        <ros>
          <namespace>/lidar</namespace>
          <remapping>~/out:=scan</remapping>
        </ros>
        <output_type>sensor_msgs/LaserScan</output_type>
        <frame_name>lidar_link</frame_name>
      </plugin>
    </sensor>
  </gazebo>

  <!-- Gazebo physics properties -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
    <mu1>0.2</mu1>
    <mu2>0.2</mu2>
    <kp>1000000.0</kp>
    <kd>1.0</kd>
  </gazebo>
</robot>
```

## Code Example: Physics Configuration in World Files

Here's an example of configuring physics properties in a Gazebo world file:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="default">
    <!-- Physics engine configuration -->
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0.0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>

    <!-- Include a ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include sun for lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Define a simple box model -->
    <model name="box">
      <pose>0 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.166667</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.166667</iyy>
            <iyz>0</iyz>
            <izz>0.166667</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

## Step-by-Step Practical Tutorial

### Configuring Sensors and Physics in Simulation

1. **Create a sensor configuration package**:
   ```bash
   cd ~/ros2_ws/src
   ros2 pkg create --build-type ament_cmake robot_sensors_config --dependencies rclpy std_msgs sensor_msgs
   ```

2. **Create the directory structure**:
   ```bash
   cd robot_sensors_config
   mkdir -p urdf config launch worlds
   ```

3. **Create a robot model with sensors** (`urdf/sensor_robot.urdf`):
   ```xml
   <!-- Use the sensor configuration URDF example above -->
   ```

4. **Create a custom world file** (`worlds/sensor_test.world`):
   ```xml
   <!-- Use the world file example above with modifications -->
   <?xml version="1.0" ?>
   <sdf version="1.7">
     <world name="sensor_test_world">
       <!-- Physics engine configuration -->
       <physics name="1ms" type="ode">
         <max_step_size>0.001</max_step_size>
         <real_time_factor>1.0</real_time_factor>
         <real_time_update_rate>1000</real_time_update_rate>
         <gravity>0 0 -9.8</gravity>
         <ode>
           <solver>
             <type>quick</type>
             <iters>10</iters>
             <sor>1.3</sor>
           </solver>
           <constraints>
             <cfm>0.0</cfm>
             <erp>0.2</erp>
             <contact_max_correcting_vel>100</contact_max_correcting_vel>
             <contact_surface_layer>0.001</contact_surface_layer>
           </constraints>
         </ode>
       </physics>

       <!-- Include a ground plane -->
       <include>
         <uri>model://ground_plane</uri>
       </include>

       <!-- Include sun for lighting -->
       <include>
         <uri>model://sun</uri>
       </include>

       <!-- Add some objects for sensor testing -->
       <model name="wall_1">
         <pose>3 0 1 0 0 0</pose>
         <link name="wall_link">
           <collision name="collision">
             <geometry>
               <box>
                 <size>0.1 6 2</size>
               </box>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <box>
                 <size>0.1 6 2</size>
               </box>
             </geometry>
             <material>
               <ambient>0.5 0.5 0.5 1</ambient>
               <diffuse>0.8 0.8 0.8 1</diffuse>
             </material>
           </visual>
           <inertial>
             <mass>100.0</mass>
             <inertia>
               <ixx>100</ixx>
               <ixy>0</ixy>
               <ixz>0</ixz>
               <iyy>100</iyy>
               <iyz>0</iyz>
               <izz>100</izz>
             </inertia>
           </inertial>
         </link>
       </model>
     </world>
   </sdf>
   ```

5. **Create a launch file for the simulation** (`launch/sensor_simulation.launch.py`):
   ```python
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
   from launch.launch_description_sources import PythonLaunchDescriptionSource
   from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
   from launch_ros.actions import Node
   from launch_ros.substitutions import FindPackageShare
   import os
   from ament_index_python.packages import get_package_share_directory


   def generate_launch_description():
       # Declare launch arguments
       use_sim_time = LaunchConfiguration('use_sim_time', default='true')
       world = LaunchConfiguration('world', default='sensor_test.world')

       # Get package share directory
       pkg_share = get_package_share_directory('robot_sensors_config')
       default_world_path = os.path.join(pkg_share, 'worlds', 'sensor_test.world')
       robot_urdf_path = os.path.join(pkg_share, 'urdf', 'sensor_robot.urdf')

       return LaunchDescription([
           # Declare launch arguments
           DeclareLaunchArgument(
               'use_sim_time',
               default_value='true',
               description='Use simulation (Gazebo) clock if true'),

           DeclareLaunchArgument(
               'world',
               default_value=default_world_path,
               description='Full path to world file to load'),

           # Start Gazebo server
           IncludeLaunchDescription(
               PythonLaunchDescriptionSource([
                   PathJoinSubstitution([
                       FindPackageShare('gazebo_ros'),
                       'launch',
                       'gzserver.launch.py'
                   ])
               ]),
               launch_arguments={
                   'world': world,
                   'verbose': 'false',
               }.items()
           ),

           # Start Gazebo client
           IncludeLaunchDescription(
               PythonLaunchDescriptionSource([
                   PathJoinSubstitution([
                       FindPackageShare('gazebo_ros'),
                       'launch',
                       'gzclient.launch.py'
                   ])
               ]),
           ),

           # Robot state publisher
           Node(
               package='robot_state_publisher',
               executable='robot_state_publisher',
               name='robot_state_publisher',
               output='screen',
               parameters=[{
                   'use_sim_time': use_sim_time,
                   'robot_description': open(robot_urdf_path).read()
               }]),

           # Spawn robot in Gazebo
           Node(
               package='gazebo_ros',
               executable='spawn_entity.py',
               arguments=[
                   '-topic', 'robot_description',
                   '-entity', 'sensor_robot',
                   '-x', '0', '-y', '0', '-z', '0.2'
               ],
               output='screen'),
       ])
   ```

6. **Update package.xml**:
   ```xml
   <?xml version="1.0"?>
   <?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
   <package format="3">
     <name>robot_sensors_config</name>
     <version>0.0.0</version>
     <description>Package for sensor and physics configuration examples</description>
     <maintainer email="user@example.com">User</maintainer>
     <license>Apache-2.0</license>

     <buildtool_depend>ament_cmake</buildtool_depend>

     <depend>rclpy</depend>
     <depend>std_msgs</depend>
     <depend>sensor_msgs</depend>
     <depend>robot_state_publisher</depend>
     <depend>gazebo_ros</depend>
     <depend>gazebo_plugins</depend>

     <export>
       <build_type>ament_cmake</build_type>
     </export>
   </package>
   ```

7. **Update CMakeLists.txt**:
   ```cmake
   cmake_minimum_required(VERSION 3.8)
   project(robot_sensors_config)

   if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
     add_compile_options(-Wall -Wextra -Wpedantic)
   endif()

   # Find dependencies
   find_package(ament_cmake REQUIRED)
   find_package(rclpy REQUIRED)
   find_package(std_msgs REQUIRED)
   find_package(sensor_msgs REQUIRED)
   find_package(robot_state_publisher REQUIRED)
   find_package(gazebo_ros REQUIRED)
   find_package(gazebo_plugins REQUIRED)

   # Install files
   install(DIRECTORY
     launch
     urdf
     worlds
     config
     DESTINATION share/${PROJECT_NAME}
   )

   ament_package()
   ```

8. **Build the package**:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select robot_sensors_config
   ```

9. **Source the workspace**:
   ```bash
   source install/setup.bash
   ```

10. **Launch the simulation with sensors**:
    ```bash
    ros2 launch robot_sensors_config sensor_simulation.launch.py
    ```

11. **In another terminal, check the sensor topics**:
    ```bash
    ros2 topic list | grep -E "(camera|imu|scan)"
    ```

12. **View sensor data**:
    ```bash
    # Camera images
    ros2 run image_view image_view _image:=/camera/image_raw

    # IMU data
    ros2 topic echo /imu/data

    # LIDAR scans
    ros2 topic echo /lidar/scan
    ```

## Summary

This chapter explored the simulation of sensors and physics in Gazebo, which are critical for realistic robot simulation. We covered how to configure various sensor types, model sensor noise and uncertainty, and set up physics properties for accurate simulation.

Proper sensor and physics configuration is essential for effective sim-to-real transfer, allowing algorithms developed in simulation to work effectively on real robots. Understanding these concepts enables the creation of realistic testing environments for robotic systems.

## Mini-Quiz

1. Which physics engines are supported by Gazebo?
   - A) ODE only
   - B) Bullet only
   - C) ODE, Bullet, and DART
   - D) Custom engine only

2. Which sensor type would be best for 3D mapping?
   - A) IMU
   - B) 2D LIDAR
   - C) 3D LIDAR or stereo camera
   - D) GPS

3. What does the max_step_size parameter control in physics configuration?
   - A) Maximum speed of the robot
   - B) Time step for physics simulation
   - C) Maximum sensor range
   - D) Update rate of sensors

5. Why is it important to model sensor noise in simulation?
   - A) It makes the simulation look more realistic
   - B) It helps algorithms work better in the real world
   - C) It increases simulation speed
   - D) It reduces computational requirements

**Answers**: 1-C, 2-B, 3-C, 4-B, 5-B