---
title: ROS2 Packages and Launch Files
sidebar_label: 04 - ROS2 Packages and Launch Files
---

# ROS2 Packages and Launch Files

## Learning Objectives

By the end of this chapter, you will be able to:
- Create and structure ROS2 packages using different build systems
- Understand the organization and purpose of package directories
- Create and configure launch files for starting multiple nodes
- Use parameters and configuration files in ROS2 packages
- Debug common issues with package dependencies and launch files
- Implement best practices for package organization and launch file design

## Introduction

ROS2 packages are the fundamental building blocks of ROS2 applications. They organize related functionality into manageable units and provide a standardized way to share code. Launch files complement packages by enabling the coordinated startup of multiple nodes with specific configurations.

Understanding packages and launch files is crucial for developing complex robotic systems where multiple nodes need to work together. This chapter will guide you through creating well-structured packages and launch files that follow ROS2 best practices.

## Core Concepts

### ROS2 Packages

A ROS2 package is a container that organizes related functionality:
- Contains source code, launch files, configuration files, and documentation
- Has a unique name within the workspace
- Includes a `package.xml` file with metadata and dependencies
- Follows specific directory conventions

### Package Build Systems

ROS2 supports multiple build systems:
- **ament_cmake**: For C++ packages using CMake
- **ament_python**: For Python packages using setuptools
- **ament_package**: For packages without source code (pure resource packages)

### Launch Files

Launch files provide a way to:
- Start multiple nodes simultaneously
- Configure parameters for nodes
- Set up remappings
- Define conditional execution
- Organize complex robot applications

### Package Directory Structure

```
my_robot_package/
├── CMakeLists.txt          # Build configuration (for CMake packages)
├── package.xml             # Package metadata and dependencies
├── setup.py                # Python setup configuration (for Python packages)
├── setup.cfg               # Python installation configuration
├── my_robot_package/       # Python module directory
│   ├── __init__.py
│   └── my_module.py
├── launch/                 # Launch files
│   └── my_robot_launch.py
├── config/                 # Configuration files
│   └── my_robot_params.yaml
├── src/                    # Source files (C++ packages)
├── include/                # Header files (C++ packages)
└── test/                   # Test files
```

## Architecture Diagram

![Architecture Diagram](/img/ch4-ad.svg)

<!-- ```mermaid
graph TB
    subgraph "ROS2 Workspace"
        A[Package A] --/> B[Package B]
        A -/-> C[Package C]
        B -/-> D[Launch File 1]
        C -/-> D
        D -/-> E[Running Nodes]
    end

    subgraph "Package Structure"
        F[package.xml]
        G[setup.py/CMakeLists.txt]
        H[Source Code]
        I[Launch Files]
        J[Config Files]
        F -/> G
        G -/> H
        H -/> I
        I --/> J
    end
``` -->

## Flow Diagram

![Architecture Diagram](/img/ch4-flow.svg)

<!-- ```mermaid
sequenceDiagram
    participant Dev as Developer
    participant Pkg as Package Creation
    participant Build as Build System
    participant Launch as Launch System
    participant Nodes as Running Nodes

    Dev->>Pkg: Create package structure
    Pkg->>Build: Build package
    Dev->>Launch: Execute launch file
    Launch->>Nodes: Start multiple nodes
    Nodes->>Dev: Provide functionality
``` -->

## Code Example: Python Package Structure

Here's an example of a Python ROS2 package structure:

`package.xml`:
```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_bringup</name>
  <version>0.0.0</version>
  <description>Package to bring up my robot</description>
  <maintainer email="user@example.com">User</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

`setup.py`:
```python
from setuptools import setup
import os
from glob import glob

package_name = 'my_robot_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Include all config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='Package to bring up my robot',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_controller = my_robot_bringup.robot_controller:main',
        ],
    },
)
```

## Code Example: Launch File

Here's an example of a launch file that starts multiple nodes:

`launch/my_robot_launch.py`:
```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    # Get package share directory
    pkg_share = get_package_share_directory('my_robot_bringup')

    # Load parameters from YAML file
    params_file = os.path.join(pkg_share, 'config', 'my_robot_params.yaml')

    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'),

        # Robot controller node
        Node(
            package='my_robot_bringup',
            executable='robot_controller',
            name='robot_controller',
            parameters=[params_file, {'use_sim_time': use_sim_time}],
            output='screen'),

        # Sensor processing node
        Node(
            package='my_robot_bringup',
            executable='sensor_processor',
            name='sensor_processor',
            parameters=[params_file, {'use_sim_time': use_sim_time}],
            output='screen'),

        # Navigation node
        Node(
            package='my_robot_bringup',
            executable='navigator',
            name='navigator',
            parameters=[params_file, {'use_sim_time': use_sim_time}],
            output='screen'),
    ])
```

## Configuration File Example

`config/my_robot_params.yaml`:
```yaml
/**:
  ros__parameters:
    use_sim_time: false
    robot_name: "my_robot"
    update_rate: 50.0
    linear_velocity_limit: 1.0
    angular_velocity_limit: 1.0

robot_controller:
  ros__parameters:
    max_acceleration: 2.0
    control_frequency: 50.0

sensor_processor:
  ros__parameters:
    sensor_timeout: 0.1
    data_buffer_size: 100
```

## Step-by-Step Practical Tutorial

### Creating a Complete ROS2 Package with Launch Files

1. **Create a new ROS2 package**:
   ```bash
   cd ~/ros2_ws/src
   ros2 pkg create --build-type ament_python my_robot_bringup --dependencies rclpy std_msgs sensor_msgs
   ```

2. **Navigate to the package directory**:
   ```bash
   cd my_robot_bringup
   ```

3. **Create the main module directory**:
   ```bash
   mkdir my_robot_bringup
   touch my_robot_bringup/__init__.py
   ```

4. **Create a simple robot controller node** (`my_robot_bringup/robot_controller.py`):
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String


   class RobotController(Node):

       def __init__(self):
           super().__init__('robot_controller')
           self.publisher_ = self.create_publisher(String, 'robot_status', 10)
           timer_period = 1.0  # seconds
           self.timer = self.create_timer(timer_period, self.timer_callback)
           self.i = 0

       def timer_callback(self):
           msg = String()
           msg.data = f'Robot status: operational - {self.i}'
           self.publisher_.publish(msg)
           self.get_logger().info(f'Publishing: "{msg.data}"')
           self.i += 1


   def main(args=None):
       rclpy.init(args=args)
       robot_controller = RobotController()
       rclpy.spin(robot_controller)
       robot_controller.destroy_node()
       rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

5. **Create launch and config directories**:
   ```bash
   mkdir launch config
   ```

6. **Create the launch file** (`launch/my_robot_launch.py`):
   ```python
   # Use the launch file example above
   ```

7. **Create the config file** (`config/my_robot_params.yaml`):
   ```yaml
   # Use the config file example above
   ```

8. **Update the setup.py file** to include data files and entry points:
   ```python
   # Use the setup.py example above with the correct data_files section
   ```

9. **Build the package**:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select my_robot_bringup
   ```

10. **Source the workspace**:
    ```bash
    source install/setup.bash
    ```

11. **Run the launch file**:
    ```bash
    ros2 launch my_robot_bringup my_robot_launch.py
    ```

## Summary

This chapter covered ROS2 packages and launch files, which are essential for organizing and deploying robotic applications. Packages provide a standardized way to organize code, while launch files enable the coordinated startup of complex systems with multiple nodes.

Well-structured packages with proper dependencies and launch files make robotic applications modular, reusable, and easier to maintain. Understanding these concepts is fundamental to building scalable robotic systems.

## Mini-Quiz

1. What is the primary purpose of a ROS2 package?
   - A) To store configuration files only
   - B) To organize related functionality into manageable units
   - C) To compile C++ code only
   - D) To provide visualization tools

2. Which build type should be used for a Python-based ROS2 package?
   - A) ament_cmake
   - B) ament_python
   - C) ament_package
   - D) colcon_build

3. What is the main advantage of using launch files?
   - A) They make code run faster
   - B) They allow starting multiple nodes with one command
   - C) They reduce memory usage
   - D) They provide better security

4. Which file contains package metadata and dependencies?
   - A) CMakeLists.txt
   - B) setup.py
   - C) package.xml
   - D) requirements.txt

5. What does the `/**:` syntax mean in a YAML parameter file?
   - A) It applies to all nodes
   - B) It's a comment
   - C) It defines a new namespace
   - D) It's an error in syntax

**Answers**: 1-B, 2-B, 3-B, 4-C, 5-A