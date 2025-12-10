---
title: Robotics Labs
sidebar_label: 23 - Robotics Labs
---

# Robotics Labs

## Learning Objectives

By the end of this chapter, you will be able to:
- Design and implement hands-on robotics laboratory exercises
- Apply theoretical concepts to practical robotics experiments
- Integrate multiple robotics subsystems in laboratory settings
- Evaluate and analyze robotics system performance through experiments
- Troubleshoot common robotics hardware and software issues
- Document and report laboratory findings effectively

## Introduction

Robotics laboratories provide essential hands-on experience where theoretical concepts meet practical implementation. These lab exercises bridge the gap between classroom learning and real-world robotics applications, allowing students and practitioners to experiment with physical robots, test algorithms, and validate theoretical models in controlled environments.

Laboratory exercises in robotics typically involve multiple subsystems including perception, planning, control, and actuation. This chapter provides guidance on designing effective robotics labs that integrate concepts from ROS2, simulation, NVIDIA Isaac, and Vision-Language-Action (VLA) models, allowing learners to build comprehensive understanding through practical experience.

## Core Concepts

### Laboratory Design Principles

- **Progressive Complexity**: Labs should start with basic concepts and gradually increase in complexity
- **Hands-On Learning**: Emphasis on practical implementation rather than theoretical analysis only
- **System Integration**: Exercises that combine multiple robotics subsystems
- **Reproducible Experiments**: Well-defined procedures that can be replicated

### Lab Components

- **Hardware Setup**: Robot platforms, sensors, and actuator systems
- **Software Environment**: ROS2, simulation tools, and development frameworks
- **Experimental Procedures**: Step-by-step instructions for conducting experiments
- **Data Collection**: Methods for measuring and recording system performance
- **Analysis Tools**: Techniques for interpreting experimental results

### Safety Considerations

- **Physical Safety**: Protecting humans and equipment during experiments
- **Operational Safety**: Safe robot operation protocols
- **Emergency Procedures**: Protocols for handling unexpected situations
- **Equipment Protection**: Safeguarding expensive robotics hardware

### Assessment and Evaluation

- **Performance Metrics**: Quantitative measures of system performance
- **Qualitative Assessment**: Observational evaluation of robot behavior
- **Documentation Requirements**: Proper recording of experimental procedures and results
- **Troubleshooting Skills**: Ability to diagnose and resolve common issues

## Architecture Diagram

![Flow Diagram](/img/ch23-ad.svg)
<!-- 
```mermaid
graph TB
    subgraph "Lab Environment"
        A[Robot Platform]
        B[Sensor Systems]
        C[Actuator Systems]
        D[Computing Hardware]
    end

    subgraph "Software Stack"
        E[ROS2 Framework]
        F[Perception Modules]
        G[Planning Algorithms]
        H[Control Systems]
    end

    subgraph "Lab Infrastructure"
        I[Workstations]
        J[Network Infrastructure]
        K[Safety Systems]
        L[Monitoring Tools]
    end

    subgraph "Experimental Design"
        M[Lab Procedures]
        N[Data Collection]
        O[Analysis Tools]
        P[Documentation]
    end

    A -/-> E
    B -/-> E
    C -/-> E
    D -/-> E
    E -/-> F
    E -/-> G
    E -/-> H
    I -/-> E
    J -/-> E
    K -/-> A
    L -/-> A
    M -/-> A
    N -/-> A
    O -/-> N
    P -/-> M
``` -->

## Flow Diagram

![Flow Diagram](/img/ch23-flow.svg)

<!-- ```mermaid
sequenceDiagram
    participant Student as Student
    participant Setup as Lab Setup
    participant Execute as Experiment Execution
    participant Observe as Observation
    participant Analyze as Data Analysis
    participant Report as Report Generation

    Student->>Setup: Prepare lab environment
    Setup->>Execute: Execute experimental procedure
    Execute->>Observe: Monitor robot behavior
    Observe->>Analyze: Analyze collected data
    Analyze->>Report: Generate lab report
    Report->>Student: Review and learn
    Student->>Setup: Iterate with improvements
``` -->

## Code Example: Lab Exercise Framework

Here's an example framework for implementing robotics lab exercises:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu
from geometry_msgs.msg import Twist, Pose, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float64
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from tf2_ros import TransformListener, Buffer
import numpy as np
import time
import threading
import queue
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import csv


class LabState(Enum):
    """States for lab execution"""
    SETUP = "setup"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


class LabExperimentType(Enum):
    """Types of lab experiments"""
    BASIC_MOBILITY = "basic_mobility"
    SENSOR_FUSION = "sensor_fusion"
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PERCEPTION = "perception"
    VLA_INTEGRATION = "vla_integration"


@dataclass
class LabMetrics:
    """Metrics collected during lab experiments"""
    timestamp: float
    experiment_type: LabExperimentType
    success_rate: float = 0.0
    execution_time: float = 0.0
    accuracy: float = 0.0
    efficiency: float = 0.0
    error_count: int = 0
    data_points: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.data_points is None:
            self.data_points = []


@dataclass
class LabConfiguration:
    """Configuration for lab experiments"""
    experiment_name: str
    experiment_type: LabExperimentType
    duration: float  # seconds
    success_criteria: Dict[str, float]
    safety_limits: Dict[str, float]
    required_equipment: List[str]
    evaluation_metrics: List[str]


class RoboticsLabFramework(Node):
    """
    Framework for implementing robotics laboratory exercises
    """
    def __init__(self):
        super().__init__('robotics_lab_framework')

        # Initialize parameters
        self.declare_parameter('lab_experiment_type', 'basic_mobility')
        self.declare_parameter('lab_duration', 300.0)  # 5 minutes default
        self.declare_parameter('data_collection_rate', 10.0)  # Hz
        self.declare_parameter('enable_visualization', True)
        self.declare_parameter('enable_logging', True)

        # Get parameters
        experiment_type_str = self.get_parameter('lab_experiment_type').value
        self.experiment_type = LabExperimentType(experiment_type_str)
        self.lab_duration = self.get_parameter('lab_duration').value
        self.data_collection_rate = self.get_parameter('data_collection_rate').value
        self.enable_visualization = self.get_parameter('enable_visualization').value
        self.enable_logging = self.get_parameter('enable_logging').value

        # Initialize components
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Lab state management
        self.lab_state = LabState.SETUP
        self.lab_start_time = None
        self.lab_end_time = None
        self.current_experiment: Optional[LabConfiguration] = None
        self.metrics = LabMetrics(
            timestamp=time.time(),
            experiment_type=self.experiment_type
        )

        # Data collection
        self.data_buffer = queue.Queue(maxsize=1000)
        self.visualization_markers = MarkerArray()

        # Experiment control
        self.experiment_thread = None
        self.experiment_running = False

        # Create publishers
        self.status_pub = self.create_publisher(String, '/lab/status', 10)
        self.metrics_pub = self.create_publisher(Float64, '/lab/metrics', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/lab/markers', 10)
        self.command_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Create subscribers
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)

        # Create timers
        self.data_collection_timer = self.create_timer(
            1.0 / self.data_collection_rate, self.collect_data)
        self.monitoring_timer = self.create_timer(1.0, self.monitor_experiment)
        self.visualization_timer = self.create_timer(0.1, self.publish_visualization)

        # Initialize lab
        self.setup_lab_environment()

        self.get_logger().info(
            f'Robotics Lab Framework initialized for {self.experiment_type.value}'
        )

    def setup_lab_environment(self):
        """
        Setup the laboratory environment
        """
        try:
            # Configure safety limits
            self.safety_limits = {
                'max_velocity': 0.5,
                'min_distance': 0.3,
                'max_acceleration': 1.0,
                'max_current': 10.0  # for actuators
            }

            # Configure success criteria
            self.success_criteria = {
                'accuracy_threshold': 0.95,
                'completion_time': self.lab_duration * 0.8,
                'error_limit': 3
            }

            # Initialize data structures
            self.robot_state = {
                'position': np.array([0.0, 0.0, 0.0]),
                'velocity': np.array([0.0, 0.0, 0.0]),
                'orientation': np.array([0.0, 0.0, 0.0, 1.0]),
                'safety_status': 'normal'
            }

            self.lab_state = LabState.READY
            self.get_logger().info('Lab environment setup completed')

        except Exception as e:
            self.get_logger().error(f'Error setting up lab environment: {e}')
            self.lab_state = LabState.ERROR

    def start_experiment(self, config: LabConfiguration):
        """
        Start a new lab experiment
        """
        try:
            self.current_experiment = config
            self.lab_start_time = time.time()
            self.lab_end_time = self.lab_start_time + config.duration
            self.lab_state = LabState.RUNNING

            # Start experiment thread
            self.experiment_running = True
            self.experiment_thread = threading.Thread(
                target=self._run_experiment, args=(config,))
            self.experiment_thread.start()

            status_msg = String()
            status_msg.data = f'Started_experiment:{config.experiment_name}'
            self.status_pub.publish(status_msg)

            self.get_logger().info(f'Started experiment: {config.experiment_name}')

        except Exception as e:
            self.get_logger().error(f'Error starting experiment: {e}')
            self.lab_state = LabState.ERROR

    def stop_experiment(self):
        """
        Stop the current experiment
        """
        try:
            self.experiment_running = False
            if self.experiment_thread:
                self.experiment_thread.join(timeout=2.0)

            self.lab_state = LabState.COMPLETED
            self.lab_end_time = time.time()

            # Calculate final metrics
            self.calculate_final_metrics()

            status_msg = String()
            status_msg.data = f'Experiment completed. Success rate: {self.metrics.success_rate:.2f}'
            self.status_pub.publish(status_msg)

            self.get_logger().info('Experiment stopped and metrics calculated')

        except Exception as e:
            self.get_logger().error(f'Error stopping experiment: {e}')

    def _run_experiment(self, config: LabConfiguration):
        """
        Run the experiment in a separate thread
        """
        try:
            if config.experiment_type == LabExperimentType.BASIC_MOBILITY:
                self._run_mobility_experiment()
            elif config.experiment_type == LabExperimentType.NAVIGATION:
                self._run_navigation_experiment()
            elif config.experiment_type == LabExperimentType.PERCEPTION:
                self._run_perception_experiment()
            elif config.experiment_type == LabExperimentType.VLA_INTEGRATION:
                self._run_vla_integration_experiment()
            # Add other experiment types as needed

        except Exception as e:
            self.get_logger().error(f'Error in experiment execution: {e}')
            self.lab_state = LabState.ERROR

    def _run_mobility_experiment(self):
        """
        Run basic mobility experiment
        """
        # Example: Move robot in a square pattern
        waypoints = [
            (1.0, 0.0),   # Move 1m forward
            (1.0, 1.0),   # Move 1m right
            (0.0, 1.0),   # Move 1m back
            (0.0, 0.0)    # Move 1m left (return to start)
        ]

        for i, (target_x, target_y) in enumerate(waypoints):
            if not self.experiment_running:
                break

            self.get_logger().info(f'Moving to waypoint {i+1}: ({target_x}, {target_y})')

            # Simple proportional controller
            while self.experiment_running:
                current_pos = self.robot_state['position']
                dx = target_x - current_pos[0]
                dy = target_y - current_pos[1]
                distance = np.sqrt(dx**2 + dy**2)

                if distance < 0.1:  # Within 10cm of target
                    break

                # Calculate velocity command
                cmd_vel = Twist()
                cmd_vel.linear.x = min(0.3, distance * 0.5)  # Proportional control
                cmd_vel.angular.z = np.arctan2(dy, dx) * 0.5

                # Check safety
                if self._check_safety():
                    self.command_pub.publish(cmd_vel)
                else:
                    self.get_logger().warn('Safety limit exceeded, stopping')
                    break

                time.sleep(0.1)

    def _run_navigation_experiment(self):
        """
        Run navigation experiment
        """
        # Example: Navigate to random points while avoiding obstacles
        for _ in range(5):  # 5 random goals
            if not self.experiment_running:
                break

            # Generate random goal
            goal_x = np.random.uniform(-2.0, 2.0)
            goal_y = np.random.uniform(-2.0, 2.0)

            self.get_logger().info(f'Navigating to goal: ({goal_x}, {goal_y})')

            # Simple navigation to goal
            while self.experiment_running:
                current_pos = self.robot_state['position']
                dx = goal_x - current_pos[0]
                dy = goal_y - current_pos[1]
                distance = np.sqrt(dx**2 + dy**2)

                if distance < 0.2:  # Within 20cm of goal
                    break

                # Calculate velocity command
                cmd_vel = Twist()
                cmd_vel.linear.x = min(0.4, distance * 0.8)
                cmd_vel.angular.z = np.arctan2(dy, dx) * 0.8

                # Check for obstacles
                if hasattr(self, 'obstacle_distance') and self.obstacle_distance < 0.5:
                    cmd_vel.linear.x = 0.0  # Stop if obstacle too close
                    cmd_vel.angular.z *= 0.5  # Turn to avoid

                # Check safety
                if self._check_safety():
                    self.command_pub.publish(cmd_vel)
                else:
                    self.get_logger().warn('Safety limit exceeded during navigation')
                    break

                time.sleep(0.1)

    def _run_perception_experiment(self):
        """
        Run perception experiment
        """
        # Example: Detect and track objects
        detection_count = 0
        tracking_accuracy = 0.0

        start_time = time.time()
        while self.experiment_running and (time.time() - start_time) < 30:  # 30 second experiment
            # Simulate object detection
            if hasattr(self, 'latest_image'):
                # In a real implementation, this would run object detection
                # For simulation, we'll count detections
                detection_count += 1
                tracking_accuracy += np.random.uniform(0.8, 1.0)  # Simulated accuracy

            time.sleep(0.5)  # Process every 0.5 seconds

        self.get_logger().info(f'Perception experiment: {detection_count} detections')

    def _run_vla_integration_experiment(self):
        """
        Run VLA integration experiment
        """
        # Example: Process voice commands and execute actions
        for _ in range(3):  # 3 command cycles
            if not self.experiment_running:
                break

            # Simulate voice command processing
            commands = ["move forward", "turn left", "grasp object"]
            command = np.random.choice(commands)

            self.get_logger().info(f'Processing command: {command}')

            # Execute based on command
            cmd_vel = Twist()
            if "move" in command:
                cmd_vel.linear.x = 0.3
            elif "turn" in command:
                cmd_vel.angular.z = 0.5
            # In a real system, this would integrate with VLA components

            self.command_pub.publish(cmd_vel)
            time.sleep(2.0)  # Execute for 2 seconds

    def odom_callback(self, msg: Odometry):
        """
        Handle odometry updates
        """
        try:
            self.robot_state['position'] = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ])

            # Convert quaternion to euler for orientation
            import math
            from tf_transformations import euler_from_quaternion
            orientation = msg.pose.pose.orientation
            roll, pitch, yaw = euler_from_quaternion([
                orientation.x, orientation.y, orientation.z, orientation.w
            ])
            self.robot_state['orientation'] = np.array([roll, pitch, yaw])

            self.robot_state['velocity'] = np.array([
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.angular.z
            ])

        except Exception as e:
            self.get_logger().error(f'Error in odom callback: {e}')

    def scan_callback(self, msg: LaserScan):
        """
        Handle laser scan updates
        """
        try:
            # Process laser scan for obstacle detection
            ranges = np.array(msg.ranges)
            valid_ranges = ranges[np.isfinite(ranges)]  # Remove invalid readings

            if len(valid_ranges) > 0:
                self.obstacle_distance = np.min(valid_ranges)
                self.robot_state['safety_status'] = 'normal' if self.obstacle_distance > 0.3 else 'warning'

                # Update safety state
                if self.obstacle_distance < 0.2:
                    self.robot_state['safety_status'] = 'critical'
                    if self.lab_state == LabState.RUNNING:
                        self.get_logger().warn('Obstacle too close, safety action required')

        except Exception as e:
            self.get_logger().error(f'Error in scan callback: {e}')

    def imu_callback(self, msg: Imu):
        """
        Handle IMU updates
        """
        try:
            # Process IMU data for stability and safety
            linear_accel = np.sqrt(
                msg.linear_acceleration.x**2 +
                msg.linear_acceleration.y**2 +
                msg.linear_acceleration.z**2
            )

            if linear_accel > 5.0:  # High acceleration detected
                self.robot_state['safety_status'] = 'warning'
                self.get_logger().warn(f'High acceleration detected: {linear_accel:.2f}')

        except Exception as e:
            self.get_logger().error(f'Error in IMU callback: {e}')

    def image_callback(self, msg: Image):
        """
        Handle image updates
        """
        try:
            # Store latest image for processing
            self.latest_image = msg

            # In a real implementation, this would process the image
            # for perception experiments

        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')

    def collect_data(self):
        """
        Collect data for metrics calculation
        """
        if self.lab_state != LabState.RUNNING:
            return

        try:
            current_time = time.time()
            if self.lab_start_time:
                elapsed_time = current_time - self.lab_start_time

                # Collect relevant data points
                data_point = {
                    'timestamp': current_time,
                    'elapsed_time': elapsed_time,
                    'position': self.robot_state['position'].tolist(),
                    'velocity': self.robot_state['velocity'].tolist(),
                    'safety_status': self.robot_state['safety_status'],
                    'obstacle_distance': getattr(self, 'obstacle_distance', float('inf'))
                }

                if not self.data_buffer.full():
                    self.data_buffer.put(data_point)
                else:
                    # Remove oldest if buffer full
                    try:
                        self.data_buffer.get_nowait()
                        self.data_buffer.put(data_point)
                    except queue.Empty:
                        pass

        except Exception as e:
            self.get_logger().error(f'Error in data collection: {e}')

    def calculate_metrics(self) -> LabMetrics:
        """
        Calculate current lab metrics
        """
        try:
            # Calculate from collected data
            data_points = []
            temp_buffer = []

            # Get all data points from buffer
            while not self.data_buffer.empty():
                try:
                    data_points.append(self.data_buffer.get_nowait())
                except queue.Empty:
                    break

            # Put data back in buffer
            for dp in data_points:
                if not self.data_buffer.full():
                    self.data_buffer.put(dp)
                temp_buffer.append(dp)

            if not temp_buffer:
                return self.metrics

            # Calculate metrics
            positions = np.array([dp['position'] for dp in temp_buffer])
            velocities = np.array([dp['velocity'] for dp in temp_buffer])

            # Success rate: percentage of time robot was in safe state
            safe_states = [dp for dp in temp_buffer if dp['safety_status'] == 'normal']
            success_rate = len(safe_states) / len(temp_buffer) if temp_buffer else 0.0

            # Accuracy: how close to desired positions (for mobility experiments)
            if len(positions) > 1:
                avg_velocity = np.mean(np.linalg.norm(velocities, axis=1))
                efficiency = avg_velocity / self.safety_limits['max_velocity']

                self.metrics = LabMetrics(
                    timestamp=time.time(),
                    experiment_type=self.experiment_type,
                    success_rate=success_rate,
                    execution_time=len(temp_buffer) / self.data_collection_rate,
                    accuracy=0.0,  # Would be calculated based on specific experiment
                    efficiency=efficiency,
                    error_count=len([dp for dp in temp_buffer if dp['safety_status'] == 'critical']),
                    data_points=temp_buffer
                )

            # Publish metrics
            metrics_msg = Float64()
            metrics_msg.data = float(self.metrics.success_rate)
            self.metrics_pub.publish(metrics_msg)

            return self.metrics

        except Exception as e:
            self.get_logger().error(f'Error calculating metrics: {e}')
            return self.metrics

    def calculate_final_metrics(self):
        """
        Calculate final metrics at experiment completion
        """
        try:
            # Get all remaining data
            all_data = []
            while not self.data_buffer.empty():
                try:
                    all_data.append(self.data_buffer.get_nowait())
                except queue.Empty:
                    break

            if not all_data:
                return

            # Calculate comprehensive metrics
            total_time = self.lab_end_time - self.lab_start_time if self.lab_start_time else 0
            success_states = [d for d in all_data if d['safety_status'] == 'normal']

            final_success_rate = len(success_states) / len(all_data) if all_data else 0.0

            # Update metrics
            self.metrics.success_rate = final_success_rate
            self.metrics.execution_time = total_time

            # Log results
            self.get_logger().info(
                f'Final Metrics - Success Rate: {final_success_rate:.2f}, '
                f'Execution Time: {total_time:.2f}s, '
                f'Errors: {self.metrics.error_count}'
            )

            # Save to file
            self._save_experiment_results()

        except Exception as e:
            self.get_logger().error(f'Error calculating final metrics: {e}')

    def _save_experiment_results(self):
        """
        Save experiment results to file
        """
        try:
            filename = f"lab_results_{self.experiment_type.value}_{int(time.time())}.json"

            results = {
                'experiment_type': self.experiment_type.value,
                'start_time': self.lab_start_time,
                'end_time': self.lab_end_time,
                'duration': self.metrics.execution_time,
                'success_rate': self.metrics.success_rate,
                'error_count': self.metrics.error_count,
                'efficiency': self.metrics.efficiency,
                'data_points_count': len(self.metrics.data_points)
            }

            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)

            self.get_logger().info(f'Experiment results saved to {filename}')

        except Exception as e:
            self.get_logger().error(f'Error saving results: {e}')

    def _check_safety(self) -> bool:
        """
        Check if current robot state is within safety limits
        """
        try:
            # Check velocity limits
            vel_magnitude = np.linalg.norm(self.robot_state['velocity'])
            if vel_magnitude > self.safety_limits['max_velocity']:
                return False

            # Check obstacle distance (if available)
            if hasattr(self, 'obstacle_distance'):
                if self.obstacle_distance < self.safety_limits['min_distance']:
                    return False

            return True

        except Exception as e:
            self.get_logger().error(f'Error in safety check: {e}')
            return False

    def monitor_experiment(self):
        """
        Monitor experiment status and safety
        """
        try:
            # Check if experiment should end
            if (self.lab_state == LabState.RUNNING and
                self.lab_end_time and
                time.time() > self.lab_end_time):
                self.stop_experiment()

            # Publish status
            status_msg = String()
            status_msg.data = f'State: {self.lab_state.value}, ' \
                             f'Experiment: {self.current_experiment.experiment_name if self.current_experiment else "None"}, ' \
                             f'Running: {self.experiment_running}'
            self.status_pub.publish(status_msg)

            # Calculate and log metrics periodically
            if self.lab_state == LabState.RUNNING:
                metrics = self.calculate_metrics()
                self.get_logger().debug(
                    f'Current metrics - Success: {metrics.success_rate:.2f}, '
                    f'Errors: {metrics.error_count}'
                )

        except Exception as e:
            self.get_logger().error(f'Error in monitoring: {e}')

    def publish_visualization(self):
        """
        Publish visualization markers for lab environment
        """
        if not self.enable_visualization:
            return

        try:
            marker_array = MarkerArray()
            current_time = self.get_clock().now()

            # Create markers for robot path
            if hasattr(self, 'robot_state'):
                path_marker = Marker()
                path_marker.header.frame_id = 'map'
                path_marker.header.stamp = current_time.to_msg()
                path_marker.ns = 'robot_path'
                path_marker.id = 0
                path_marker.type = Marker.SPHERE
                path_marker.action = Marker.ADD

                path_marker.pose.position.x = float(self.robot_state['position'][0])
                path_marker.pose.position.y = float(self.robot_state['position'][1])
                path_marker.pose.position.z = 0.0
                path_marker.pose.orientation.w = 1.0

                path_marker.scale.x = 0.1
                path_marker.scale.y = 0.1
                path_marker.scale.z = 0.1

                path_marker.color.r = 1.0
                path_marker.color.g = 0.0
                path_marker.color.b = 0.0
                path_marker.color.a = 0.8

                marker_array.markers.append(path_marker)

            self.marker_pub.publish(marker_array)

        except Exception as e:
            self.get_logger().error(f'Error in visualization: {e}')

    def destroy_node(self):
        """
        Clean up resources when node is destroyed
        """
        self.get_logger().info('Cleaning up Robotics Lab Framework')
        if self.experiment_running:
            self.stop_experiment()
        super().destroy_node()


class LabExperimentManager:
    """
    Manager for organizing and running lab experiments
    """
    def __init__(self):
        self.available_experiments = {}
        self.completed_experiments = []
        self.current_experiment = None

    def register_experiment(self, name: str, config: LabConfiguration):
        """
        Register a new experiment configuration
        """
        self.available_experiments[name] = config
        print(f"Registered experiment: {name}")

    def get_experiment(self, name: str) -> Optional[LabConfiguration]:
        """
        Get a registered experiment configuration
        """
        return self.available_experiments.get(name)

    def run_experiment(self, framework: RoboticsLabFramework, name: str) -> bool:
        """
        Run a registered experiment
        """
        config = self.get_experiment(name)
        if not config:
            print(f"Experiment {name} not found")
            return False

        try:
            framework.start_experiment(config)
            self.current_experiment = config
            return True
        except Exception as e:
            print(f"Error running experiment {name}: {e}")
            return False

    def create_basic_mobility_config(self) -> LabConfiguration:
        """
        Create configuration for basic mobility lab
        """
        return LabConfiguration(
            experiment_name="Basic Mobility Test",
            experiment_type=LabExperimentType.BASIC_MOBILITY,
            duration=120.0,  # 2 minutes
            success_criteria={
                'accuracy_threshold': 0.90,
                'completion_time': 100.0,
                'max_errors': 2
            },
            safety_limits={
                'max_velocity': 0.5,
                'min_distance': 0.3
            },
            required_equipment=['robot_base', 'laser_scanner', 'odometry'],
            evaluation_metrics=['success_rate', 'accuracy', 'efficiency']
        )

    def create_navigation_config(self) -> LabConfiguration:
        """
        Create configuration for navigation lab
        """
        return LabConfiguration(
            experiment_name="Navigation Test",
            experiment_type=LabExperimentType.NAVIGATION,
            duration=300.0,  # 5 minutes
            success_criteria={
                'accuracy_threshold': 0.85,
                'completion_time': 240.0,
                'max_errors': 5
            },
            safety_limits={
                'max_velocity': 0.4,
                'min_distance': 0.4
            },
            required_equipment=['robot_base', 'laser_scanner', 'odometry', 'mapping_system'],
            evaluation_metrics=['success_rate', 'path_efficiency', 'obstacle_avoidance']
        )

    def create_perception_config(self) -> LabConfiguration:
        """
        Create configuration for perception lab
        """
        return LabConfiguration(
            experiment_name="Perception Test",
            experiment_type=LabExperimentType.PERCEPTION,
            duration=180.0,  # 3 minutes
            success_criteria={
                'detection_accuracy': 0.80,
                'tracking_stability': 0.90,
                'max_errors': 3
            },
            safety_limits={
                'max_velocity': 0.2,  # Slow for perception
                'min_distance': 0.5
            },
            required_equipment=['camera', 'robot_base', 'lighting'],
            evaluation_metrics=['detection_rate', 'accuracy', 'processing_time']
        )


def create_lab_curriculum():
    """
    Create a complete lab curriculum
    """
    curriculum = {
        'level_1': {
            'name': 'Basic Robotics Concepts',
            'experiments': [
                'Motor Control',
                'Basic Sensing',
                'Simple Navigation'
            ],
            'duration_hours': 6,
            'prerequisites': ['Basic programming', 'Introduction to ROS']
        },
        'level_2': {
            'name': 'Intermediate Robotics',
            'experiments': [
                'Sensor Fusion',
                'Path Planning',
                'Object Manipulation'
            ],
            'duration_hours': 12,
            'prerequisites': ['Level 1 completion', 'Basic control theory']
        },
        'level_3': {
            'name': 'Advanced Robotics Applications',
            'experiments': [
                'SLAM Implementation',
                'Vision-Based Control',
                'Human-Robot Interaction'
            ],
            'duration_hours': 18,
            'prerequisites': ['Level 2 completion', 'Advanced programming']
        }
    }

    return curriculum


def main(args=None):
    """
    Main function for robotics lab framework
    """
    rclpy.init(args=args)

    try:
        # Create lab framework
        lab_framework = RoboticsLabFramework()

        # Create experiment manager
        exp_manager = LabExperimentManager()

        # Register experiments
        exp_manager.register_experiment('basic_mobility', exp_manager.create_basic_mobility_config())
        exp_manager.register_experiment('navigation', exp_manager.create_navigation_config())
        exp_manager.register_experiment('perception', exp_manager.create_perception_config())

        # Example: Run basic mobility experiment
        exp_manager.run_experiment(lab_framework, 'basic_mobility')

        # Spin the framework
        rclpy.spin(lab_framework)

    except KeyboardInterrupt:
        pass
    finally:
        if 'lab_framework' in locals():
            lab_framework.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Step-by-Step Practical Tutorial

### Implementing Robotics Lab Exercises

1. **Create a lab package**:
   ```bash
   cd ~/ros2_ws/src
   ros2 pkg create --build-type ament_python robotics_labs_examples --dependencies rclpy std_msgs sensor_msgs geometry_msgs nav_msgs visualization_msgs cv_bridge tf2_ros
   ```

2. **Navigate to the package directory**:
   ```bash
   cd robotics_labs_examples
   ```

3. **Create the main module directory**:
   ```bash
   mkdir robotics_labs_examples
   touch robotics_labs_examples/__init__.py
   ```

4. **Create the lab framework implementation** (`robotics_labs_examples/lab_framework.py`):
   ```python
   # Use the lab framework code example above
   ```

5. **Create a configuration file** (`config/lab_config.yaml`):
   ```yaml
   robotics_lab_framework:
     ros__parameters:
       # Lab parameters
       lab_experiment_type: "basic_mobility"
       lab_duration: 300.0
       data_collection_rate: 10.0
       enable_visualization: true
       enable_logging: true

       # Safety parameters
       max_velocity: 0.5
       min_distance: 0.3
       max_acceleration: 1.0

       # Evaluation parameters
       success_threshold: 0.8
       error_limit: 5

       # Topic configuration
       odom_topic: "/odom"
       scan_topic: "/scan"
       image_topic: "/camera/image_raw"
       cmd_vel_topic: "/cmd_vel"
   ```

6. **Create lab exercise files** (`robotics_labs_examples/lab_exercises/`):
   ```bash
   mkdir robotics_labs_examples/lab_exercises
   ```

7. **Create a basic mobility lab** (`robotics_labs_examples/lab_exercises/basic_mobility.py`):
   ```python
   """
   Basic Mobility Lab Exercise
   """
   import rclpy
   from rclpy.node import Node
   from geometry_msgs.msg import Twist
   from nav_msgs.msg import Odometry
   from sensor_msgs.msg import LaserScan
   import numpy as np
   import time


   class BasicMobilityLab(Node):
       """
       Basic mobility lab exercise: Move robot in geometric patterns
       """
       def __init__(self):
           super().__init__('basic_mobility_lab')

           # Robot state
           self.current_position = np.array([0.0, 0.0])
           self.current_orientation = 0.0
           self.safety_distance = 0.5

           # Create publishers and subscribers
           self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
           self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
           self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

           # Create timer for control loop
           self.control_timer = self.create_timer(0.1, self.control_loop)

           self.get_logger().info('Basic Mobility Lab initialized')

       def odom_callback(self, msg):
           """
           Handle odometry updates
           """
           self.current_position[0] = msg.pose.pose.position.x
           self.current_position[1] = msg.pose.pose.position.y

       def scan_callback(self, msg):
           """
           Handle laser scan for obstacle detection
           """
           if len(msg.ranges) > 0:
               min_distance = min([r for r in msg.ranges if r > 0])
               if min_distance < self.safety_distance:
                   self.get_logger().warn(f'Obstacle detected: {min_distance:.2f}m')

       def control_loop(self):
           """
           Main control loop for mobility exercise
           """
           # Example: Move in a square pattern
           cmd_vel = Twist()
           cmd_vel.linear.x = 0.3  # Move forward at 0.3 m/s
           cmd_vel.angular.z = 0.0

           # In a real exercise, this would implement the specific mobility task
           self.cmd_vel_pub.publish(cmd_vel)

       def execute_square_pattern(self):
           """
           Execute movement in a square pattern
           """
           waypoints = [
               (1.0, 0.0),   # Move 1m forward
               (1.0, 1.0),   # Move 1m right
               (0.0, 1.0),   # Move 1m back
               (0.0, 0.0)    # Move 1m left
           ]

           for target_x, target_y in waypoints:
               self.move_to_position(target_x, target_y)

       def move_to_position(self, target_x, target_y):
           """
           Move robot to target position
           """
           while True:
               dx = target_x - self.current_position[0]
               dy = target_y - self.current_position[1]
               distance = np.sqrt(dx**2 + dy**2)

               if distance < 0.1:  # Within 10cm
                   break

               cmd_vel = Twist()
               cmd_vel.linear.x = min(0.3, distance * 0.5)
               cmd_vel.angular.z = np.arctan2(dy, dx) * 0.5

               self.cmd_vel_pub.publish(cmd_vel)
               time.sleep(0.1)


   def main(args=None):
       rclpy.init(args=args)
       lab_node = BasicMobilityLab()

       try:
           rclpy.spin(lab_node)
       except KeyboardInterrupt:
           pass
       finally:
           lab_node.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

8. **Create a perception lab** (`robotics_labs_examples/lab_exercises/perception_lab.py`):
   ```python
   """
   Perception Lab Exercise
   """
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import Image, LaserScan
   from std_msgs.msg import String
   from cv_bridge import CvBridge
   import numpy as np
   import cv2


   class PerceptionLab(Node):
       """
       Perception lab exercise: Object detection and tracking
       """
       def __init__(self):
           super().__init__('perception_lab')

           # Initialize CV bridge
           self.bridge = CvBridge()

           # Create publishers and subscribers
           self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
           self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
           self.result_pub = self.create_publisher(String, '/perception/results', 10)

           self.get_logger().info('Perception Lab initialized')

       def image_callback(self, msg):
           """
           Process camera images for object detection
           """
           try:
               # Convert ROS image to OpenCV
               cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

               # Simple color-based object detection (example)
               hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

               # Define range for red color
               lower_red = np.array([0, 100, 100])
               upper_red = np.array([10, 255, 255])
               mask1 = cv2.inRange(hsv, lower_red, upper_red)

               lower_red = np.array([170, 100, 100])
               upper_red = np.array([180, 255, 255])
               mask2 = cv2.inRange(hsv, lower_red, upper_red)

               mask = mask1 + mask2

               # Find contours
               contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

               if contours:
                   # Find largest contour
                   largest_contour = max(contours, key=cv2.contourArea)
                   if cv2.contourArea(largest_contour) > 500:  # Minimum area threshold
                       # Calculate center of object
                       M = cv2.moments(largest_contour)
                       if M["m00"] != 0:
                           cx = int(M["m10"] / M["m00"])
                           cy = int(M["m01"] / M["m00"])

                           result_msg = String()
                           result_msg.data = f'Red object detected at ({cx}, {cy})'
                           self.result_pub.publish(result_msg)

                           self.get_logger().info(f'Red object detected at ({cx}, {cy})')

           except Exception as e:
               self.get_logger().error(f'Error processing image: {e}')

       def scan_callback(self, msg):
           """
           Process laser scan for object detection
           """
           # In a real implementation, this would detect objects using LIDAR
           pass


   def main(args=None):
       rclpy.init(args=args)
       lab_node = PerceptionLab()

       try:
           rclpy.spin(lab_node)
       except KeyboardInterrupt:
           pass
       finally:
           lab_node.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

9. **Create launch directory**:
   ```bash
   mkdir launch
   ```

10. **Create a launch file** (`launch/robotics_labs_example.launch.py`):
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
        pkg_share = get_package_share_directory('robotics_labs_examples')
        config_file = os.path.join(pkg_share, 'config', 'lab_config.yaml')

        return LaunchDescription([
            # Declare launch arguments
            DeclareLaunchArgument(
                'use_sim_time',
                default_value='false',
                description='Use simulation time if true'),

            # Lab framework node
            Node(
                package='robotics_labs_examples',
                executable='robotics_labs_examples.lab_framework',
                name='robotics_lab_framework',
                parameters=[
                    config_file,
                    {'use_sim_time': use_sim_time}
                ],
                output='screen'
            ),

            # Basic mobility lab
            Node(
                package='robotics_labs_examples',
                executable='robotics_labs_examples.lab_exercises.basic_mobility',
                name='basic_mobility_lab',
                parameters=[{'use_sim_time': use_sim_time}],
                output='screen'
            ),

            # Perception lab
            Node(
                package='robotics_labs_examples',
                executable='robotics_labs_examples.lab_exercises.perception_lab',
                name='perception_lab',
                parameters=[{'use_sim_time': use_sim_time}],
                output='screen'
            )
        ])
    ```

11. **Update setup.py**:
    ```python
    from setuptools import setup
    import os
    from glob import glob

    package_name = 'robotics_labs_examples'

    setup(
        name=package_name,
        version='0.0.0',
        packages=[package_name, f'{package_name}.lab_exercises'],
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
        description='Robotics lab examples for education',
        license='Apache-2.0',
        tests_require=['pytest'],
        entry_points={
            'console_scripts': [
                'lab_framework = robotics_labs_examples.lab_framework:main',
                'basic_mobility_lab = robotics_labs_examples.lab_exercises.basic_mobility:main',
                'perception_lab = robotics_labs_examples.lab_exercises.perception_lab:main',
            ],
        },
    )
    ```

12. **Build the package**:
    ```bash
    cd ~/ros2_ws
    colcon build --packages-select robotics_labs_examples
    ```

13. **Source the workspace**:
    ```bash
    source install/setup.bash
    ```

14. **Run the robotics lab example**:
    ```bash
    ros2 launch robotics_labs_examples robotics_labs_example.launch.py
    ```

15. **Monitor lab status**:
    ```bash
    ros2 topic echo /lab/status
    ros2 topic echo /lab/metrics
    ```

## Summary

This chapter provided a comprehensive framework for implementing robotics laboratory exercises that integrate concepts from throughout the textbook. We covered the design principles for effective lab exercises, safety considerations, and practical implementation techniques for various types of robotics experiments.

Robotics labs are essential for bridging the gap between theoretical knowledge and practical application, allowing students to gain hands-on experience with real robotic systems. The framework provided enables the creation of progressive, safe, and educational laboratory experiences.

## Mini-Quiz

1. What is a key principle in designing robotics lab exercises?
   - A) Focus only on software
   - B) Progressive complexity from basic to advanced
   - C) Only use simulation
   - D) Focus on theory only

2. Which safety consideration is important in robotics labs?
   - A) Only software safety
   - B) Physical safety for humans and equipment
   - C) Only network security
   - D) Only data protection

3. What should be included in lab assessment?
   - A) Only written tests
   - B) Performance metrics and qualitative assessment
   - C) Only final results
   - D) Only attendance

4. Why is data collection important in lab exercises?
   - A) Only for reports
   - B) To measure performance and validate theories
   - C) Only for documentation
   - D) Only for publications

5. What is the purpose of lab documentation?
   - A) Only for instructors
   - B) To record procedures and results for learning
   - C) Only for safety
   - D) Only for equipment tracking

**Answers**: 1-B, 2-B, 3-B, 4-B, 5-B