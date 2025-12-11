---
title: ROS2 Parameters and Configuration
sidebar_label: 05 - ROS2 Parameters and Configuration
---

# ROS2 Parameters and Configuration

## Learning Objectives

By the end of this chapter, you will be able to:
- Define and use ROS2 parameters for runtime configuration
- Create and manage parameter files in YAML format
- Implement dynamic parameter handling in nodes
- Use parameter declarations and callbacks
- Organize configuration for complex robotic systems
- Debug common issues with parameter management

## Introduction

Parameters in ROS2 provide a flexible way to configure nodes at runtime without recompiling code. They allow for easy adjustment of robot behaviors, sensor settings, and algorithm parameters. This chapter explores the parameter system, configuration files, and best practices for managing complex robotic system configurations.

Parameters are crucial for creating adaptable robotic systems that can operate in different environments or with different hardware configurations. Understanding parameter management is essential for developing production-ready robotic applications.

## Core Concepts

### ROS2 Parameters

Parameters are named values that can be:
- Set at launch time from configuration files
- Modified at runtime through command-line tools
- Declared and validated within nodes
- Used to control node behavior dynamically

### Parameter Types

ROS2 supports these parameter types:
- `bool`: Boolean values (true/false)
- `int`: Integer values
- `double`: Floating-point values
- `string`: Text values
- `byte_array`: Array of bytes
- `bool_array`: Array of boolean values
- `integer_array`: Array of integer values
- `double_array`: Array of floating-point values
- `string_array`: Array of string values

### Parameter Namespaces

Parameters can be organized using namespaces:
- Node-specific parameters: `node_name.parameter_name`
- Global parameters: accessible across the system
- Hierarchical organization for complex systems

### Parameter Files

YAML files provide a standardized way to store parameter configurations:
- Human-readable format
- Version control friendly
- Supports complex nested structures
- Can be loaded at launch time

## Architecture Diagram

<div style={{ textAlign: "center" }}>
  <img
    src="/img/ch5-flow.png"
    alt="Architecture Diagram"
    style={{ maxWidth: "800px", width: "80%" }}
  />
</div>

<!-- ```mermaid
graph TB
    subgraph "Parameter Server"
        A[Parameter Server]
    end

    subgraph "Node A"
        B[Node A]
        B1[Declared Parameters]
    end

    subgraph "Node B"
        C[Node B]
        C1[Declared Parameters]
    end

    subgraph "Configuration"
        D[YAML Config Files]
        E[Command Line]
        F[Parameter Client Tools]
    end

    D -/-> A
    E -/-> A
    F -/-> A
    A -/-> B1
    A -/-> C1
    B1 -/-> B
    C1 -/-> C
``` -->

## Flow Diagram

<div style={{ textAlign: "center" }}>
  <img
    src="/img/ch5-ad.svg"
    alt="Flow Diagram"
    style={{ maxWidth: "700px", width: "70%" }}
  />
</div>

<!-- ```mermaid
sequenceDiagram
    participant Config as Configuration
    participant Node as ROS2 Node
    participant Param as Parameter System

    Config->>Param: Load YAML parameters
    Param->>Node: Provide parameters at startup
    Node->>Node: Declare parameters with defaults
    Node->>Param: Update parameter during runtime
    Config->>Param: Change parameter via command line
    Param->>Node: Notify parameter change
``` -->

## Code Example: Parameter Declaration and Usage

Here's an example of a ROS2 node that declares and uses parameters:

```python
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_system_default


class ParameterNode(Node):

    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values and descriptions
        self.declare_parameter('robot_name', 'default_robot',
                             ParameterDescriptor(description='Name of the robot'))
        self.declare_parameter('max_velocity', 1.0,
                             ParameterDescriptor(description='Maximum velocity in m/s'))
        self.declare_parameter('sensors_enabled', True,
                             ParameterDescriptor(description='Enable sensor processing'))
        self.declare_parameter('topics_list', ['sensor1', 'sensor2'],
                             ParameterDescriptor(description='List of sensor topics'))

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.sensors_enabled = self.get_parameter('sensors_enabled').value
        self.topics_list = self.get_parameter('topics_list').value

        # Create parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Timer to periodically check parameters
        self.timer = self.create_timer(2.0, self.check_parameters)

        self.get_logger().info(f'Initialized with robot_name: {self.robot_name}')

    def parameter_callback(self, params):
        """
        Callback function to handle parameter changes
        """
        for param in params:
            if param.name == 'max_velocity' and param.type_ == Parameter.Type.DOUBLE:
                if param.value > 5.0:
                    return SetParametersResult(successful=False, reason='Max velocity too high')
                else:
                    self.max_velocity = param.value
                    self.get_logger().info(f'Updated max_velocity to: {param.value}')
            elif param.name == 'robot_name' and param.type_ == Parameter.Type.STRING:
                self.robot_name = param.value
                self.get_logger().info(f'Updated robot_name to: {param.value}')

        return SetParametersResult(successful=True)

    def check_parameters(self):
        """
        Periodically check parameter values
        """
        current_max_vel = self.get_parameter('max_velocity').value
        if current_max_vel != self.max_velocity:
            self.max_velocity = current_max_vel
            self.get_logger().info(f'Parameter changed externally to: {current_max_vel}')


def main(args=None):
    rclpy.init(args=args)
    parameter_node = ParameterNode()

    try:
        rclpy.spin(parameter_node)
    except KeyboardInterrupt:
        pass
    finally:
        parameter_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Parameter YAML Configuration File

Here's an example of a parameter configuration file:

```yaml
# my_robot_params.yaml
/**:  # Applies to all nodes
  ros__parameters:
    use_sim_time: false
    global_timeout: 5.0

robot_controller:
  ros__parameters:
    robot_name: "my_robot"
    max_linear_velocity: 1.0
    max_angular_velocity: 1.5
    acceleration_limit: 2.0
    deceleration_limit: 3.0

sensor_processor:
  ros__parameters:
    sensor_timeout: 0.1
    data_buffer_size: 100
    publish_frequency: 10.0
    sensor_offsets:
      x: 0.1
      y: 0.0
      z: 0.2

navigation:
  ros__parameters:
    planner_frequency: 5.0
    controller_frequency: 20.0
    recovery_enabled: true
    oscillation_timeout: 10.0
    oscillation_distance: 0.5
    global_frame: "map"
    robot_base_frame: "base_link"
    transform_tolerance: 0.2
    min_vel_theta: 0.4
    max_vel_theta: 1.0
    min_in_place_vel_theta: 0.4
    max_in_place_vel_theta: 2.0
    theta_stopped_vel: 0.1
    min_vel_x: 0.0
    max_vel_x: 0.5
    min_vel_y: -0.5
    max_vel_y: 0.5
    y_stopped_vel: 0.1
```

## Code Example: Loading Parameters in Launch Files

Here's how to load parameters in a launch file:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Declare launch arguments
    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Get package share directory
    pkg_share = get_package_share_directory('my_robot_bringup')

    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(pkg_share, 'config', 'my_robot_params.yaml'),
            description='Full path to the parameters file to use'),

        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'),

        # Robot controller node with parameters
        Node(
            package='my_robot_bringup',
            executable='robot_controller',
            name='robot_controller',
            parameters=[params_file, {'use_sim_time': use_sim_time}],
            output='screen',
            respawn=True),  # Restart if the node dies
    ])
```

## Step-by-Step Practical Tutorial

### Creating a Parameter-Configured ROS2 Node

1. **Create a new ROS2 package**:
   ```bash
   cd ~/ros2_ws/src
   ros2 pkg create --build-type ament_python my_robot_params --dependencies rclpy std_msgs
   ```

2. **Navigate to the package directory**:
   ```bash
   cd my_robot_params
   ```

3. **Create the main module directory**:
   ```bash
   mkdir my_robot_params
   touch my_robot_params/__init__.py
   ```

4. **Create a parameter-based node** (`my_robot_params/param_controller.py`):
   ```python
   import rclpy
   from rclpy.node import Node
   from rclpy.parameter import Parameter
   from rclpy.exceptions import ParameterNotDeclaredException
   from rclpy.parameter_service import ParameterService
   from rcl_interfaces.msg import ParameterDescriptor
   from rcl_interfaces.srv import SetParameters


   class ParamController(Node):

       def __init__(self):
           super().__init__('param_controller')

           # Declare parameters with descriptions
           self.declare_parameter('robot_name', 'default_robot',
                                ParameterDescriptor(description='Name of the robot'))
           self.declare_parameter('max_velocity', 1.0,
                                ParameterDescriptor(description='Maximum velocity in m/s'))
           self.declare_parameter('sensors_enabled', True,
                                ParameterDescriptor(description='Enable sensor processing'))

           # Get initial parameter values
           self.update_parameters()

           # Create parameter change timer
           self.timer = self.create_timer(3.0, self.check_parameters)

           self.get_logger().info(f'Initialized with robot_name: {self.robot_name}')

       def update_parameters(self):
           """Update internal values from parameters"""
           try:
               self.robot_name = self.get_parameter('robot_name').value
               self.max_velocity = self.get_parameter('max_velocity').value
               self.sensors_enabled = self.get_parameter('sensors_enabled').value
           except ParameterNotDeclaredException as e:
               self.get_logger().error(f'Parameter not declared: {e}')

       def check_parameters(self):
           """Periodically check for parameter changes"""
           self.update_parameters()
           self.get_logger().info(f'Current config - Robot: {self.robot_name}, Max Vel: {self.max_velocity}, Sensors: {self.sensors_enabled}')


   def main(args=None):
       rclpy.init(args=args)
       param_controller = ParamController()

       try:
           rclpy.spin(param_controller)
       except KeyboardInterrupt:
           pass
       finally:
           param_controller.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

5. **Create config directory**:
   ```bash
   mkdir config
   ```

6. **Create a parameter configuration file** (`config/robot_params.yaml`):
   ```yaml
   param_controller:
     ros__parameters:
       robot_name: "my_param_robot"
       max_velocity: 2.5
       sensors_enabled: true
       debug_mode: false
   ```

7. **Update setup.py** with entry points:
   ```python
   from setuptools import setup
   import os
   from glob import glob

   package_name = 'my_robot_params'

   setup(
       name=package_name,
       version='0.0.0',
       packages=[package_name],
       data_files=[
           ('share/ament_index/resource_index/packages',
               ['resource/' + package_name]),
           ('share/' + package_name, ['package.xml']),
           (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
       ],
       install_requires=['setuptools'],
       zip_safe=True,
       maintainer='User',
       maintainer_email='user@example.com',
       description='Package for parameter configuration examples',
       license='Apache-2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'param_controller = my_robot_params.param_controller:main',
           ],
       },
   )
   ```

8. **Build the package**:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select my_robot_params
   ```

9. **Source the workspace**:
   ```bash
   source install/setup.bash
   ```

10. **Run the node**:
    ```bash
    ros2 run my_robot_params param_controller
    ```

11. **In another terminal, change parameters at runtime**:
    ```bash
    ros2 param set /param_controller robot_name "new_robot_name"
    ros2 param set /param_controller max_velocity 3.0
    ```

12. **List current parameters**:
    ```bash
    ros2 param list
    ros2 param dump /param_controller
    ```

## Summary

This chapter covered ROS2 parameters and configuration, which provide flexible runtime configuration for robotic systems. Parameters allow nodes to be configured without recompilation and can be managed through YAML files, launch files, or command-line tools.

Proper parameter management is essential for creating adaptable robotic systems that can operate in different environments or with different hardware configurations. Understanding parameter systems is crucial for developing production-ready robotic applications.

## Mini-Quiz

1. What is the main advantage of using parameters in ROS2?
   - A) They make code run faster
   - B) They allow runtime configuration without recompilation
   - C) They reduce memory usage
   - D) They provide better security

2. Which parameter type would you use for a list of sensor names?
   - A) string
   - B) string_array
   - C) string_list
   - D) array_string

3. How do you declare a parameter with a default value in a ROS2 node?
   - A) `self.param('param_name', default)`
   - B) `self.declare_parameter('param_name', default)`
   - C) `declare_parameter('param_name', default)`
   - D) `self.add_parameter('param_name', default)`

4. What does the `/**:` syntax mean in a YAML parameter file?
   - A) It applies the parameters to all nodes
   - B) It's a comment
   - C) It defines a new namespace
   - D) It's an error in syntax

5. Which command can be used to set a parameter at runtime?
   - A) ros2 set param
   - B) ros2 param set
   - C) ros2 change param
   - D) ros2 update param

**Answers**: 1-B, 2-B, 3-B, 4-A, 5-B