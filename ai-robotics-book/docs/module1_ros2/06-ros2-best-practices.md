---
title: ROS2 Best Practices
sidebar_label: 06 - ROS2 Best Practices
---

# ROS2 Best Practices

## Learning Objectives

By the end of this chapter, you will be able to:
- Apply software engineering best practices to ROS2 development
- Design efficient and maintainable ROS2 node architectures
- Implement proper error handling and logging strategies
- Follow ROS2 community standards and conventions
- Optimize ROS2 applications for performance and reliability
- Debug and profile ROS2 applications effectively

## Introduction

ROS2 best practices encompass the accumulated knowledge and experience of the robotics community in developing robust, maintainable, and efficient robotic systems. These practices span from code organization and architecture to performance optimization and debugging techniques.

Following best practices is essential for creating production-ready robotic applications that can be maintained, extended, and deployed reliably. This chapter consolidates the most important practices for ROS2 development based on real-world experience.

## Core Concepts

### Node Design Principles

- **Single Responsibility**: Each node should have one clear purpose
- **Modularity**: Design nodes to be reusable and replaceable
- **Cohesion**: Group related functionality within a node
- **Coupling**: Minimize dependencies between nodes

### Communication Best Practices

- **Topic Design**: Use descriptive names and appropriate message types
- **QoS Configuration**: Select appropriate Quality of Service policies for your use case
- **Message Efficiency**: Design compact and efficient message structures
- **Rate Limiting**: Avoid excessive message publishing rates

### Performance Considerations

- **Threading**: Use appropriate threading models for your application
- **Memory Management**: Minimize memory allocations in critical loops
- **CPU Utilization**: Balance performance with system resource usage
- **Real-time Constraints**: Understand and implement real-time considerations when needed

### Code Quality Standards

- **Documentation**: Document all public interfaces and complex algorithms
- **Testing**: Implement unit tests, integration tests, and system tests
- **Code Review**: Establish code review processes for quality assurance
- **Version Control**: Use Git effectively with clear commit messages

## Architecture Diagram

![Architecture Diagram](/img/ch6-ad.png)

<!-- ```mermaid
graph TB
    subgraph "Best Practices Layer"
        A[Single Responsibility]
        B[Error Handling]
        C[Performance]
        D[Testing]
        E[Documentation]
        F[Security]
    end

    subgraph "ROS2 Application"
        G[Node A]
        H[Node B]
        I[Node C]
        J[DDS Middleware]
    end

    A -.-> G
    A -.-> H
    A -.-> I
    B -.-> G
    B -.-> H
    B -.-> I
    C -.-> G
    C -.-> H
    C -.-> I
    D -.-> G
    D -.-> H
    D -.-> I
    E -.-> G
    E -.-> H
    E -.-> I
    F -.-> J
    G -.-> J
    H -.-> J
    I -.-> J
``` -->

## Flow Diagram

![Architecture Diagram](/img/ch6-flow.svg)

<!-- ```mermaid
sequenceDiagram
    participant Dev as Developer
    participant Code as Code Implementation
    participant Test as Testing
    participant Deploy as Deployment
    participant Monitor as Monitoring

    Dev->>Code: Follow best practices
    Code->>Test: Include tests
    Test->>Deploy: Pass all tests
    Deploy->>Monitor: Monitor performance
    Monitor->>Dev: Provide feedback
    Dev->>Code: Refine based on feedback
``` -->

## Code Example: Well-Structured ROS2 Node

Here's an example of a ROS2 node following best practices:

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
import threading
from typing import Optional
import time


class BestPracticeNode(Node):
    """
    Example ROS2 node demonstrating best practices.

    This node follows ROS2 best practices including:
    - Proper parameter declaration and validation
    - Error handling and logging
    - Efficient message handling
    - Resource management
    - Clear documentation
    """

    def __init__(self):
        super().__init__('best_practice_node')

        # 1. Parameter declaration with validation
        self.declare_parameters(
            namespace='',
            parameters=[
                ('robot_name', 'default_robot', rclpy.ParameterDescriptor(
                    description='Name of the robot')),
                ('publish_rate', 10, rclpy.ParameterDescriptor(
                    description='Rate at which to publish messages')),
                ('sensor_timeout', 1.0, rclpy.ParameterDescriptor(
                    description='Timeout for sensor data in seconds')),
                ('debug_mode', False, rclpy.ParameterDescriptor(
                    description='Enable debug output')),
            ]
        )

        # 2. Get parameters with type safety
        self.robot_name = self.get_parameter('robot_name').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.sensor_timeout = self.get_parameter('sensor_timeout').value
        self.debug_mode = self.get_parameter('debug_mode').value

        # 3. Validate parameters
        if self.publish_rate <= 0:
            self.get_logger().error('Invalid publish_rate, using default of 10')
            self.publish_rate = 10

        # 4. Setup QoS profiles for different use cases
        # Reliable communication for critical data
        reliable_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        # Best effort for high-frequency data
        best_effort_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # 5. Create publishers and subscribers
        self.status_publisher = self.create_publisher(
            String, 'robot_status', reliable_qos)

        self.sensor_subscriber = self.create_subscription(
            LaserScan, 'laser_scan', self.sensor_callback, best_effort_qos)

        # 6. Create timer with proper rate control
        self.timer = self.create_timer(
            1.0 / self.publish_rate, self.status_timer_callback)

        # 7. Initialize state variables safely
        self.last_sensor_data_time = self.get_clock().now()
        self.sensor_data_lock = threading.Lock()
        self.latest_scan_data = None

        self.get_logger().info(
            f'BestPracticeNode initialized for robot: {self.robot_name}')

    def sensor_callback(self, msg: LaserScan):
        """
        Callback for sensor data with proper error handling.

        Args:
            msg: LaserScan message from sensor
        """
        try:
            # Validate message data
            if len(msg.ranges) == 0:
                self.get_logger().warning('Received empty laser scan')
                return

            # Protect shared data with lock
            with self.sensor_data_lock:
                self.latest_scan_data = msg
                self.last_sensor_data_time = self.get_clock().now()

            if self.debug_mode:
                self.get_logger().debug(f'Received scan with {len(msg.ranges)} ranges')

        except Exception as e:
            self.get_logger().error(f'Error in sensor_callback: {e}')

    def status_timer_callback(self):
        """
        Timer callback for publishing status messages.
        """
        try:
            # Check for sensor timeout
            current_time = self.get_clock().now()
            time_since_sensor = (current_time - self.last_sensor_data_time).nanoseconds / 1e9

            if time_since_sensor > self.sensor_timeout:
                self.get_logger().warn(f'Sensor data timeout: {time_since_sensor:.2f}s')

            # Create and publish status message
            status_msg = String()
            status_msg.data = f'{self.robot_name}: operational, sensor_age: {time_since_sensor:.2f}s'
            self.status_publisher.publish(status_msg)

            if self.debug_mode:
                self.get_logger().debug(f'Published status: {status_msg.data}')

        except Exception as e:
            self.get_logger().error(f'Error in status_timer_callback: {e}')

    def destroy_node(self):
        """
        Properly clean up resources when node is destroyed.
        """
        self.get_logger().info('Cleaning up BestPracticeNode resources')
        # Any custom cleanup code here
        super().destroy_node()


def main(args=None):
    """
    Main function with proper exception handling.
    """
    rclpy.init(args=args)

    try:
        best_practice_node = BestPracticeNode()
        rclpy.spin(best_practice_node)
    except KeyboardInterrupt:
        print('Interrupted by user')
    except Exception as e:
        print(f'Error during execution: {e}')
    finally:
        # Always clean up
        if 'best_practice_node' in locals():
            best_practice_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Launch File Best Practices

Here's an example of a well-structured launch file:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, RegisterEventHandler
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from launch.event_handlers import OnProcessExit
import os


def generate_launch_description():
    # Get package share directory
    pkg_share = get_package_share_directory('best_practices_examples')

    # Declare launch arguments with descriptions
    debug_mode_arg = DeclareLaunchArgument(
        'debug_mode',
        default_value='false',
        description='Enable debug output and tools'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    # Get launch configurations
    debug_mode = LaunchConfiguration('debug_mode')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Define nodes with proper configuration
    best_practice_node = Node(
        package='best_practices_examples',
        executable='best_practice_node',
        name='best_practice_node',
        parameters=[
            os.path.join(pkg_share, 'config', 'best_practice_params.yaml'),
            {'use_sim_time': use_sim_time},
            {'debug_mode': debug_mode}
        ],
        output='screen',
        # Restart if the node dies
        respawn=True,
        respawn_delay=2.0,
        # Additional environment variables if needed
        additional_env={'RCUTILS_COLORIZED_OUTPUT': '1'}
    )

    # Conditional nodes (only run if debug mode is enabled)
    debug_node = Node(
        package='best_practices_examples',
        executable='debug_node',
        name='debug_node',
        condition=IfCondition(debug_mode),
        parameters=[{'use_sim_time': use_sim_time}]
    )

    return LaunchDescription([
        # Launch arguments
        debug_mode_arg,
        use_sim_time_arg,

        # Environment setup
        SetEnvironmentVariable(name='RCUTILS_LOGGING_SEVERITY_THRESHOLD', value='INFO'),

        # Nodes
        best_practice_node,
        debug_node,
    ])
```

## Testing Best Practices

Here's an example of unit testing for a ROS2 node:

```python
import unittest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String
from best_practices_examples.best_practice_node import BestPracticeNode


class TestBestPracticeNode(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = BestPracticeNode()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

    def tearDown(self):
        self.node.destroy_node()

    def test_parameter_declaration(self):
        """Test that parameters are properly declared."""
        # Check that parameters exist
        self.assertTrue(self.node.has_parameter('robot_name'))
        self.assertTrue(self.node.has_parameter('publish_rate'))
        self.assertTrue(self.node.has_parameter('sensor_timeout'))
        self.assertTrue(self.node.has_parameter('debug_mode'))

    def test_parameter_values(self):
        """Test that parameters have correct default values."""
        robot_name_param = self.node.get_parameter('robot_name').value
        self.assertEqual(robot_name_param, 'default_robot')

        publish_rate_param = self.node.get_parameter('publish_rate').value
        self.assertEqual(publish_rate_param, 10)

    def test_status_publishing(self):
        """Test that status messages are published."""
        # Create a subscription to catch published messages
        received_msgs = []

        def msg_callback(msg):
            received_msgs.append(msg)

        sub = self.node.create_subscription(
            String, 'robot_status', msg_callback, 10)

        # Trigger the timer callback manually
        self.node.status_timer_callback()

        # Process any pending callbacks
        self.executor.spin_once(timeout_sec=0.1)

        # Check that a message was published
        self.assertEqual(len(received_msgs), 1)
        self.assertIn('default_robot', received_msgs[0].data)


if __name__ == '__main__':
    unittest.main()
```

## Step-by-Step Practical Tutorial

### Implementing Best Practices in a Complete Example

1. **Create a new ROS2 package for best practices examples**:
   ```bash
   cd ~/ros2_ws/src
   ros2 pkg create --build-type ament_python best_practices_examples --dependencies rclpy std_msgs sensor_msgs
   ```

2. **Navigate to the package directory**:
   ```bash
   cd best_practices_examples
   ```

3. **Create the main module directory**:
   ```bash
   mkdir best_practices_examples
   touch best_practices_examples/__init__.py
   ```

4. **Create the best practice node** (`best_practices_examples/best_practice_node.py`):
   ```python
   # Use the well-structured node code example above
   ```

5. **Create launch and config directories**:
   ```bash
   mkdir launch config
   ```

6. **Create a parameter configuration file** (`config/best_practice_params.yaml`):
   ```yaml
   best_practice_node:
     ros__parameters:
       robot_name: "best_practice_robot"
       publish_rate: 5
       sensor_timeout: 2.0
       debug_mode: false
   ```

7. **Create the launch file** (`launch/best_practices_launch.py`):
   ```python
   # Use the well-structured launch file example above
   ```

8. **Create a test directory**:
   ```bash
   mkdir test
   ```

9. **Create a test file** (`test/test_best_practice_node.py`):
   ```python
   # Use the testing example above
   ```

10. **Update setup.py** with proper configuration:
    ```python
    from setuptools import setup
    import os
    from glob import glob

    package_name = 'best_practices_examples'

    setup(
        name=package_name,
        version='0.0.0',
        packages=[package_name],
        data_files=[
            ('share/ament_index/resource_index/packages',
                ['resource/' + package_name]),
            ('share/' + package_name, ['package.xml']),
            (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
            (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        ],
        install_requires=['setuptools'],
        zip_safe=True,
        maintainer='User',
        maintainer_email='user@example.com',
        description='Package demonstrating ROS2 best practices',
        license='Apache-2.0',
        tests_require=['pytest'],
        entry_points={
            'console_scripts': [
                'best_practice_node = best_practices_examples.best_practice_node:main',
            ],
        },
    )
    ```

11. **Build the package**:
    ```bash
    cd ~/ros2_ws
    colcon build --packages-select best_practices_examples
    ```

12. **Source the workspace**:
    ```bash
    source install/setup.bash
    ```

13. **Run the best practices node**:
    ```bash
    ros2 run best_practices_examples best_practice_node
    ```

14. **Run with launch file**:
    ```bash
    ros2 launch best_practices_examples best_practices_launch.py debug_mode:=true
    ```

## Summary

This chapter covered essential ROS2 best practices that should be followed when developing robotic applications. These practices include proper node design, error handling, parameter management, testing, and documentation.

Following these best practices results in more maintainable, reliable, and efficient robotic systems. They help ensure that your ROS2 applications can be deployed in production environments and maintained over time.

## Mini-Quiz

1. What is the primary principle behind single responsibility for ROS2 nodes?
   - A) A node should handle all robot functions
   - B) Each node should have one clear purpose
   - C) A node should only publish messages
   - D) A node should only subscribe to messages

2. What does QoS stand for in ROS2?
   - A) Quality of Service
   - B) Quick Operating System
   - C) Quality Operating System
   - D) Quantum Operating Service

3. Which testing level is most important for ROS2 nodes?
   - A) Unit testing only
   - B) Integration testing only
   - C) Both unit and integration testing
   - D) No testing needed

4. What is the purpose of the `declare_parameters` method?
   - A) To publish parameters to topics
   - B) To declare multiple parameters at once with descriptions
   - C) To subscribe to parameter changes
   - D) To delete parameters

5. What should you do in the `destroy_node` method?
   - A) Nothing special
   - B) Properly clean up resources when the node is destroyed
   - C) Restart the node
   - D) Publish final messages

**Answers**: 1-B, 2-A, 3-C, 4-B, 5-B