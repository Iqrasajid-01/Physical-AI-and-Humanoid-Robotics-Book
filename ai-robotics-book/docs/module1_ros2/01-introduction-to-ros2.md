---
title: Introduction to ROS2
sidebar_label: 01 - Introduction to ROS2
---

# Introduction to ROS2

## Learning Objectives

By the end of this chapter, you will be able to:
- Define ROS2 and explain its role in robotics development
- Identify the key differences between ROS1 and ROS2
- Describe the core concepts of nodes, topics, services, and actions
- Understand the ROS2 ecosystem and its components
- Set up a basic ROS2 development environment

## Introduction

Robot Operating System 2 (ROS2) is the next-generation robotics middleware that provides libraries and tools to help software developers create robot applications. Unlike its predecessor, ROS2 is designed to be production-ready, with improved security, real-time capabilities, and support for multiple operating systems.

ROS2 has become the standard framework for developing complex robotic systems, from autonomous vehicles to humanoid robots. Its distributed architecture allows multiple processes to communicate seamlessly, making it ideal for the Physical AI and Humanoid Robotics applications we'll explore throughout this textbook.

## Core Concepts

### What is ROS2?

ROS2 is not an actual operating system, but rather a middleware framework that provides services designed for a heterogeneous computer cluster. It includes hardware abstraction, device drivers, libraries, visualizers, message-passing, package management, and more.

### Key Differences from ROS1

1. **Middleware**: ROS2 uses DDS (Data Distribution Service) as its default middleware, providing better scalability and reliability.
2. **Real-time Support**: ROS2 has built-in support for real-time systems, crucial for safety-critical robotic applications.
3. **Security**: ROS2 includes security features by design, supporting authentication, access control, and encryption.
4. **Multi-platform**: ROS2 runs on Linux, Windows, and macOS, with experimental support for real-time systems.
5. **Quality of Service (QoS)**: ROS2 provides QoS policies for tuning communication behavior based on application requirements.

### ROS2 Ecosystem Components

- **Nodes**: Processes that perform computation
- **Topics**: Named buses over which nodes exchange messages
- **Services**: Synchronous request/response communication pattern
- **Actions**: Asynchronous request/goal-based communication with feedback
- **Packages**: Software containers that organize related functionality
- **Launch files**: Configuration files to start multiple nodes at once

## Flow Diagram
![Flow Diagram](/img/ch1-flow.svg)

<!-- ```mermaid
sequenceDiagram
    participant D as DDS Middleware
    participant N1 as Node 1
    participant N2 as Node 2

    N1->>D: Publish message to topic
    D->>N2: Deliver message to subscriber
    N2->>N1: Reply to service request
``` -->

## Code Example: Simple ROS2 Publisher

Here's a basic example of a ROS2 publisher node in Python:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Step-by-Step Practical Tutorial

### Setting up Your First ROS2 Workspace

1. **Install ROS2 Humble Hawksbill** (or Iron Irwini) following the official installation guide
2. **Create a new workspace directory**:
   ```bash
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws
   ```

3. **Source the ROS2 setup**:
   ```bash
   source /opt/ros/humble/setup.bash  # or iron
   ```

4. **Build your workspace**:
   ```bash
   colcon build
   ```

5. **Source the workspace**:
   ```bash
   source install/setup.bash
   ```

6. **Run your first publisher node**:
   ```bash
   ros2 run demo_nodes_cpp talker
   ```

7. **In a new terminal, run the subscriber**:
   ```bash
   ros2 run demo_nodes_py listener
   ```

## Summary

This chapter introduced you to ROS2, the next-generation robotics middleware that forms the foundation of modern robotics development. We covered the key differences from ROS1, the core concepts of the ROS2 architecture, and walked through setting up your first ROS2 workspace.

ROS2's distributed architecture, security features, and real-time capabilities make it ideal for Physical AI and Humanoid Robotics applications. In the next chapters, we'll dive deeper into specific ROS2 concepts like nodes, topics, services, and actions.

## Mini-Quiz

1. What does DDS stand for in the context of ROS2?
   - A) Distributed Data Service
   - B) Data Distribution Service
   - C) Dynamic Data System
   - D) Distributed Database System

2. Which of the following is NOT a key difference between ROS1 and ROS2?
   - A) Real-time support
   - B) Security features
   - C) Middleware implementation
   - D) Support for only Linux systems

3. What are the four main communication patterns in ROS2?
   - A) Nodes, Topics, Services, Actions
   - B) Publishers, Subscribers, Clients, Servers
   - C) Nodes, Topics, Messages, Services
   - D) Publishers, Subscribers, Services, Actions

4. What is the purpose of Quality of Service (QoS) policies in ROS2?
   - A) To improve security
   - B) To tune communication behavior based on application requirements
   - C) To increase processing speed
   - D) To reduce memory usage

5. Which of the following is a valid ROS2 middleware implementation?
   - A) FastDDS
   - B) CycloneDDS
   - C) Both A and B
   - D) Neither A nor B

**Answers**: 1-B, 2-D, 3-A, 4-B, 5-C