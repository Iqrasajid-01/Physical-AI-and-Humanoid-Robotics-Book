---
title: Robot Modeling with URDF
sidebar_label: 08 - Robot Modeling with URDF
---

# Robot Modeling with URDF

## Learning Objectives

By the end of this chapter, you will be able to:
- Define Unified Robot Description Format (URDF) and its role in robotics
- Create complete robot models using URDF syntax
- Understand the structure of links, joints, and materials in URDF
- Integrate Gazebo-specific extensions into URDF models
- Validate and visualize URDF models in simulation
- Apply best practices for robot model design and organization

## Introduction

Unified Robot Description Format (URDF) is an XML-based format used to describe robot models in ROS and ROS2. It defines the physical and visual properties of a robot, including its links, joints, inertial properties, and visual appearance. URDF is fundamental to robotics simulation and visualization, enabling robots to be represented in simulation environments like Gazebo.

URDF models are essential for robot simulation, visualization, and control. They provide the geometric and physical information needed for collision detection, kinematic calculations, and visual rendering. Understanding URDF is crucial for anyone working with robot simulation, planning, or control systems.

## Core Concepts

### URDF Structure

A URDF file contains:
- **Links**: Rigid bodies that make up the robot structure
- **Joints**: Connections between links that define how they can move relative to each other
- **Visual**: How the link appears visually in simulation
- **Collision**: How the link interacts with other objects for collision detection
- **Inertial**: Physical properties for dynamics simulation

### Link Properties

Links define rigid bodies in the robot model:
- **Visual**: Shape, material, and appearance
- **Collision**: Collision geometry and properties
- **Inertial**: Mass, center of mass, and inertia tensor
- **Origin**: Position and orientation relative to parent

### Joint Types

URDF supports several joint types:
- **Fixed**: No movement between links
- **Revolute**: Single-axis rotation with limits
- **Continuous**: Single-axis rotation without limits
- **Prismatic**: Single-axis translation with limits
- **Planar**: Motion on a plane
- **Floating**: 6DOF movement without constraints

### Gazebo Integration

URDF models can include Gazebo-specific extensions:
- **Gazebo plugins**: For simulation functionality
- **Materials**: For visual appearance in Gazebo
- **Transmission**: For actuator integration

## Architecture Diagram

![Architecture Diagram](/img/ch8-ad.svg)

<!-- 
```mermaid
graph TB
    subgraph "URDF Robot Model"
        A[Robot Root]
        B[Link 1]
        C[Link 2]
        D[Link 3]
        E[Joint 1-2]
        F[Joint 2-3]
    end

    subgraph "URDF Components"
        G[Visual]
        H[Collision]
        I[Inertial]
        J[Origin]
    end

    subgraph "Simulation Integration"
        K[Gazebo Plugin]
        L[Transmission]
        M[Material]
    end

    A -/-> B
    B -/-> E
    E -/-> C
    C -/-> F
    F -/-> D
    B -/-> G
    B -/-> H
    B -/-> I
    B -/-> J
    C -/-> G
    C -/-> H
    C -/-> I
    C -/-> J
    D -/-> G
    D -/-> H
    D -/-> I
    D -/-> J
    B -/-> K
    C -/-> L
    D -/-> M
``` -->

## Flow Diagram

![Architecture Diagram](/img/ch7-flow.svg)

<!-- ```mermaid
sequenceDiagram
    participant Model as URDF Model
    participant Parser as URDF Parser
    participant Sim as Simulation Engine
    participant Vis as Visualization

    Model->>Parser: Load URDF file
    Parser->>Sim: Create kinematic tree
    Parser->>Vis: Create visual representation
    Sim->>Vis: Update positions during simulation
    Vis->>User: Display robot model
``` -->

## Code Example: Complete URDF Robot Model

Here's a more complex robot model in URDF format:

```xml
<?xml version="1.0"?>
<robot name="mobile_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Materials -->
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="green">
    <color rgba="0.0 0.8 0.0 1.0"/>
  </material>
  <material name="grey">
    <color rgba="0.2 0.2 0.2 1.0"/>
  </material>
  <material name="orange">
    <color rgba="1.0 0.5 0.0 1.0"/>
  </material>
  <material name="brown">
    <color rgba="0.87 0.84 0.71 1.0"/>
  </material>
  <material name="red">
    <color rgba="0.8 0.0 0.0 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.2" length="0.1"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.2" length="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.2"/>
    </inertial>
  </link>

  <!-- Left wheel -->
  <link name="left_wheel">
    <visual>
      <origin xyz="0 0 0" rpy="1.570796 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.04"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="1.570796 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Right wheel -->
  <link name="right_wheel">
    <visual>
      <origin xyz="0 0 0" rpy="1.570796 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.04"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="1.570796 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Caster wheel -->
  <link name="caster_wheel">
    <visual>
      <geometry>
        <sphere radius="0.03"/>
      </geometry>
      <material name="grey"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.03"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.15 -0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0 -0.15 -0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <joint name="caster_wheel_joint" type="fixed">
    <parent link="base_link"/>
    <child link="caster_wheel"/>
    <origin xyz="-0.15 0 -0.07" rpy="0 0 0"/>
  </joint>

  <!-- Gazebo plugins -->
  <gazebo>
    <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
      <ros>
        <namespace>/mobile_robot</namespace>
        <remapping>cmd_vel:=cmd_vel</remapping>
        <remapping>odom:=odom</remapping>
      </ros>
      <update_rate>30</update_rate>
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>0.3</wheel_separation>
      <wheel_diameter>0.1</wheel_diameter>
      <max_wheel_torque>20</max_wheel_torque>
      <max_wheel_acceleration>1.0</max_wheel_acceleration>
      <publish_odom>true</publish_odom>
      <publish_odom_tf>true</publish_odom_tf>
      <publish_wheel_tf>true</publish_wheel_tf>
      <odometry_frame>odom</odometry_frame>
      <robot_base_frame>base_link</robot_base_frame>
    </plugin>
  </gazebo>

  <!-- Gazebo materials -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="left_wheel">
    <material>Gazebo/Black</material>
  </gazebo>

  <gazebo reference="right_wheel">
    <material>Gazebo/Black</material>
  </gazebo>

  <gazebo reference="caster_wheel">
    <material>Gazebo/Grey</material>
  </gazebo>
</robot>
```

## Xacro Example: Parameterized Robot Model

Xacro (XML Macros) allows for parameterized robot models:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="mobile_robot_xacro">
  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="wheel_radius" value="0.05" />
  <xacro:property name="wheel_width" value="0.04" />
  <xacro:property name="wheel_mass" value="0.5" />
  <xacro:property name="base_radius" value="0.2" />
  <xacro:property name="base_length" value="0.1" />
  <xacro:property name="base_mass" value="5.0" />

  <!-- Macro for wheel -->
  <xacro:macro name="wheel" params="prefix *joint_origin reflect">
    <link name="${prefix}_wheel">
      <visual>
        <origin xyz="0 0 0" rpy="${M_PI/2} 0 0"/>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <material name="black"/>
      </visual>
      <collision>
        <origin xyz="0 0 0" rpy="${M_PI/2} 0 0"/>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="${wheel_mass}"/>
        <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.002"/>
      </inertial>
    </link>

    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="base_link"/>
      <child link="${prefix}_wheel"/>
      <xacro:insert_block name="joint_origin"/>
      <axis xyz="0 1 0"/>
    </joint>

    <!-- Gazebo material -->
    <gazebo reference="${prefix}_wheel">
      <material>Gazebo/Black</material>
    </gazebo>
  </xacro:macro>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="${base_radius}" length="${base_length}"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="${base_radius}" length="${base_length}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${base_mass}"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.2"/>
    </inertial>
  </link>

  <!-- Define wheels using macro -->
  <xacro:wheel prefix="left">
    <origin xyz="0 0.15 -0.05" rpy="0 0 0"/>
  </xacro:wheel>

  <xacro:wheel prefix="right">
    <origin xyz="0 -0.15 -0.05" rpy="0 0 0"/>
  </xacro:wheel>
</robot>
```

## Step-by-Step Practical Tutorial

### Creating a Complete Robot Model

1. **Create a robot description package**:
   ```bash
   cd ~/ros2_ws/src
   ros2 pkg create --build-type ament_cmake my_robot_description
   ```

2. **Create the directory structure**:
   ```bash
   cd my_robot_description
   mkdir -p urdf meshes config launch
   ```

3. **Create the main URDF file** (`urdf/mobile_robot.urdf`):
   ```xml
   <!-- Use the complete URDF robot model example above -->
   ```

4. **Create a Xacro version** (`urdf/mobile_robot.xacro`):
   ```xml
   <!-- Use the Xacro example above -->
   ```

5. **Create a launch file for visualization** (`launch/robot_state_publisher.launch.py`):
   ```python
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import Node
   from launch_ros.substitutions import FindPackageShare
   import os
   from ament_index_python.packages import get_package_share_directory


   def generate_launch_description():
       # Declare launch arguments
       urdf_model = LaunchConfiguration('urdf_model')
       use_sim_time = LaunchConfiguration('use_sim_time', default='false')

       # Get URDF file path
       pkg_share = get_package_share_directory('my_robot_description')
       default_urdf_model_path = os.path.join(pkg_share, 'urdf', 'mobile_robot.urdf')

       return LaunchDescription([
           # Declare launch arguments
           DeclareLaunchArgument(
               'urdf_model',
               default_value=default_urdf_model_path,
               description='Absolute path to robot urdf file'),

           DeclareLaunchArgument(
               'use_sim_time',
               default_value='false',
               description='Use simulation time if true'),

           # Robot state publisher
           Node(
               package='robot_state_publisher',
               executable='robot_state_publisher',
               name='robot_state_publisher',
               output='screen',
               parameters=[{
                   'use_sim_time': use_sim_time,
                   'robot_description': open(default_urdf_model_path).read()
               }]),

           # Joint state publisher (for visualization)
           Node(
               package='joint_state_publisher_gui',
               executable='joint_state_publisher_gui',
               name='joint_state_publisher_gui',
               condition=launch.conditions.IfCondition(
                   launch.substitutions.LaunchConfiguration('use_gui', default='true')
               ))
       ])
   ```

6. **Update CMakeLists.txt** to install files:
   ```cmake
   cmake_minimum_required(VERSION 3.8)
   project(my_robot_description)

   if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
     add_compile_options(-Wall -Wextra -Wpedantic)
   endif()

   # Find dependencies
   find_package(ament_cmake REQUIRED)
   find_package(rosidl_default_generators REQUIRED)

   # Install files
   install(DIRECTORY
     launch
     urdf
     meshes
     config
     DESTINATION share/${PROJECT_NAME}
   )

   ament_package()
   ```

7. **Update package.xml**:
   ```xml
   <?xml version="1.0"?>
   <?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
   <package format="3">
     <name>my_robot_description</name>
     <version>0.0.0</version>
     <description>Robot description for mobile robot</description>
     <maintainer email="user@example.com">User</maintainer>
     <license>Apache-2.0</license>

     <buildtool_depend>ament_cmake</buildtool_depend>

     <depend>robot_state_publisher</depend>
     <depend>joint_state_publisher</depend>
     <depend>joint_state_publisher_gui</depend>
     <depend>xacro</depend>

     <export>
       <build_type>ament_cmake</build_type>
     </export>
   </package>
   ```

8. **Build the package**:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select my_robot_description
   ```

9. **Source the workspace**:
   ```bash
   source install/setup.bash
   ```

10. **Launch the robot model**:
    ```bash
    ros2 launch my_robot_description robot_state_publisher.launch.py
    ```

11. **In another terminal, run RViz to visualize**:
    ```bash
    rviz2
    ```
    Add a RobotModel display and set the TF topic to visualize the robot.

## Summary

This chapter covered URDF (Unified Robot Description Format), the standard for describing robot models in ROS/ROS2. We explored the structure of URDF files, including links, joints, and visual/collision properties, and learned how to integrate them with simulation environments like Gazebo.

URDF models are fundamental to robotics simulation and control, providing the geometric and physical information needed for robot applications. Understanding URDF is essential for anyone working with robot simulation, planning, or control systems.

## Mini-Quiz

1. What does URDF stand for?
   - A) Unified Robot Design Format
   - B) Universal Robot Description Format
   - C) Unified Robot Description Format
   - D) Universal Robot Design Framework

2. Which joint type allows continuous rotation without limits?
   - A) Revolute
   - B) Prismatic
   - C) Fixed
   - D) Continuous

3. What is Xacro used for in robot modeling?
   - A) To create 3D meshes
   - B) To provide XML macros and parameterization for URDF
   - C) To simulate robot motion
   - D) To control robot hardware

4. Which tool can be used to visualize a URDF model?
   - A) RViz
   - B) Gazebo
   - C) Both A and B
   - D) Neither A nor B

**Answers**: 1-C, 2-D, 3-C, 4-B, 5-C