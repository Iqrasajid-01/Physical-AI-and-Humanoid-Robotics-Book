---
title: ROS2 Nodes and Topics
sidebar_label: 02 - ROS2 Nodes and Topics
---

# ROS2 Nodes and Topics

## Learning Objectives

By the end of this chapter, you will be able to:
- Define ROS2 nodes and explain their role in the system architecture
- Understand the publish-subscribe communication pattern using topics
- Create and implement custom message types
- Design efficient topic-based communication systems
- Debug common issues with node and topic communication

## Introduction

Nodes and topics form the backbone of ROS2's communication architecture. A node is a process that performs computation, while topics provide a mechanism for nodes to exchange messages through a publish-subscribe pattern. This chapter will explore how nodes and topics work together to enable distributed robotic systems.

Understanding nodes and topics is fundamental to building robust robotic applications. They allow different components of a robot to communicate efficiently, whether it's sensor data flowing from perception modules to planning systems, or control commands moving from decision-making algorithms to actuator interfaces.

## Core Concepts

### ROS2 Nodes

A node is an executable that uses ROS2 client library to communicate with other nodes. Nodes can:
- Publish messages to topics
- Subscribe to topics to receive messages
- Provide services
- Call services
- Send and receive actions

Nodes should be designed to perform a single function well, following the Unix philosophy of small, focused processes that work together.

### Topics and Publish-Subscribe Pattern

Topics enable asynchronous communication between nodes using a publish-subscribe pattern:
- Publishers send messages to a topic
- Subscribers receive messages from a topic
- The ROS2 middleware (DDS) handles message routing
- Multiple publishers and subscribers can exist for the same topic

### Message Types

Messages are the data structures that flow between nodes. They are defined using `.msg` files and can contain:
- Primitive types (int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64, string, bool)
- Arrays of primitive types
- Other message types (nested messages)

## Architecture Diagram
![Flow Diagram](/img/architecture-diagram.png)
<!-- <div style={{ textAlign: "center" }}>
  <img
    src="/img/architecture-diagram.png"
    alt="Architecture Diagram"
    style={{ maxWidth: "400px", width: "80%" }}
  />
</div> -->

<!-- ```mermaid
graph TB
    subgraph "Node A - Publisher"
        A1[Node Process]
        A2[Publisher Component]
        A1 -\> A2
    end

    subgraph "DDS Middleware"
        DDS[(DDS)]
    end

    subgraph "Node B - Subscriber"
        B2[Subscriber Component]
        B1[Node Process]
        B2 -\> B1
    end

    A2 -\>|Topic: /sensor_data| DDS
    DDS -\>|Message: sensor_msgs/Image| B2
``` -->

## Flow Diagram

![Flow Diagram](/img/ch2-flow.svg)

<!-- <div style={{ textAlign: "center" }}>
  <img
    src="/img/ch2-flow.svg"
    alt="Flow Diagram"
    style={{ maxWidth: "600px", width: "80%" }}
  />
</div> -->

<!-- ```mermaid
sequenceDiagram
    participant P as Publisher Node
    participant DDS as DDS Middleware
    participant S as Subscriber Node

    P->>DDS: Publish message to /topic
    DDS->>S: Deliver message to subscriber
    Note over P,S: Multiple subscribers can receive the same message
``` -->

## Code Example: Node with Publisher and Subscriber

Here's an example of a ROS2 node that both publishes and subscribes to topics:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class TalkerListenerNode(Node):

    def __init__(self):
        super().__init__('talker_listener_node')

        # Create publisher
        self.publisher_ = self.create_publisher(String, 'chatter', 10)

        # Create subscriber
        self.subscription = self.create_subscription(
            String,
            'input',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Timer for publishing
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.i = 0

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)
        # Echo the message back
        response_msg = String()
        response_msg.data = f'Echo: {msg.data}'
        self.publisher_.publish(response_msg)
        self.get_logger().info('Publishing: "%s"' % response_msg.data)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    node = TalkerListenerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Custom Message Example

To create a custom message type, create a `.msg` file in your package's `msg` directory:

```
# CustomMessage.msg
string name
int32 id
float64[] values
geometry_msgs/Point position
```

Then use it in your code:

```python
from my_package_msgs.msg import CustomMessage

# In your node:
def create_custom_message(self):
    msg = CustomMessage()
    msg.name = "Robot1"
    msg.id = 1
    msg.values = [1.0, 2.0, 3.0]
    msg.position.x = 1.0
    msg.position.y = 2.0
    msg.position.z = 0.0
    return msg
```

## Step-by-Step Practical Tutorial

### Creating a Node with Custom Publisher and Subscriber

1. **Create a new ROS2 package**:
   ```bash
   cd ~/ros2_ws/src
   ros2 pkg create --build-type ament_python my_robot_nodes
   ```

2. **Navigate to the package directory**:
   ```bash
   cd my_robot_nodes
   ```

3. **Create the Python script** (`my_robot_nodes/my_robot_nodes/talker_listener.py`):
   ```python
   # Use the code example above
   ```

4. **Make the script executable**:
   ```bash
   chmod +x my_robot_nodes/talker_listener.py
   ```

5. **Update the setup.py file** to include the entry point:
   ```python
   entry_points={
       'console_scripts': [
           'talker_listener = my_robot_nodes.talker_listener:main',
       ],
   },
   ```

6. **Build the package**:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select my_robot_nodes
   ```

7. **Source the workspace**:
   ```bash
   source install/setup.bash
   ```

8. **Run the node**:
   ```bash
   ros2 run my_robot_nodes talker_listener
   ```

9. **In a new terminal, publish messages to test**:
   ```bash
   ros2 topic pub /input std_msgs/String "data: 'Hello from command line'"
   ```

## Summary

This chapter covered the fundamental concepts of ROS2 nodes and topics, which form the backbone of ROS2's communication system. We explored the publish-subscribe pattern, learned how to create nodes that both publish and subscribe to topics, and examined custom message types.

Nodes and topics enable the distributed architecture that makes ROS2 powerful for robotics applications. By following the publish-subscribe pattern, we can create modular systems where different components can communicate without tight coupling.

## Mini-Quiz

1. What is the primary communication pattern used by ROS2 topics?
   - A) Request-Response
   - B) Publish-Subscribe
   - C) Peer-to-Peer
   - D) Client-Server

2. Which of the following can a ROS2 node do?
   - A) Publish messages to topics
   - B) Subscribe to topics
   - C) Provide and call services
   - D) All of the above

3. What happens when multiple nodes publish to the same topic?
   - A) Only the first publisher's messages are received
   - B) Messages from all publishers are received by subscribers
   - C) The system crashes
   - D) Messages are merged together

4. What is the purpose of Quality of Service (QoS) profiles for topics?
   - A) To define security settings only
   - B) To tune communication behavior like reliability and durability
   - C) To limit the number of publishers
   - D) To encrypt message content

5. Which command can be used to list all available topics?
   - A) ros2 show topics
   - B) ros2 list topics
   - C) ros2 topic list
   - D) ros2 show all

**Answers**: 1-B, 2-D, 3-B, 4-B, 5-C