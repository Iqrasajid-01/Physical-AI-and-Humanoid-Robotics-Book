---
title: ROS2 Services and Actions
sidebar_label: 03 - ROS2 Services and Actions
---

# ROS2 Services and Actions

## Learning Objectives

By the end of this chapter, you will be able to:
- Define ROS2 services and explain their synchronous request-response pattern
- Create and implement custom service types
- Design efficient service-based communication systems
- Understand the differences between services and actions
- Implement action servers and clients for long-running tasks
- Debug common issues with service and action communication

## Introduction

While topics provide asynchronous communication through the publish-subscribe pattern, services and actions provide synchronous and goal-based communication patterns respectively. Services enable request-response interactions, while actions are designed for long-running tasks with feedback and status updates.

Services and actions are essential for robotics applications where certain operations require guaranteed delivery and response. For example, requesting the current robot pose, saving a map, or executing a complex navigation task all benefit from these communication patterns.

## Core Concepts

### ROS2 Services

Services provide a synchronous request-response communication pattern:
- A service client sends a request to a service server
- The server processes the request and sends back a response
- The client waits for the response (blocking call)
- Services are ideal for operations that have a clear start and end

### Service Types

Services are defined using `.srv` files that specify:
- Request message fields
- Response message fields
- Separated by a `---` line

### ROS2 Actions

Actions are designed for long-running tasks and provide:
- Goal requests with parameters
- Continuous feedback during execution
- Result responses upon completion
- Ability to cancel ongoing operations

Actions are ideal for tasks like navigation, manipulation, or calibration that take time and need to report progress.

## Architecture Diagram

![Architecture Diagram](/img/ch3-ad.svg)

<!-- ```mermaid
graph TB
    subgraph "Service Communication"
        A[Service Client]
        B[Service Server]
        A -.->|"Request: /service"| B
        B -.->|"Response: /service"| A
    end
    subgraph "Action Communication"
        C[Action Client]
        D[Action Server]
        C -.->|"Goal: /action/goal"| D
        D -.->|"Feedback: /action/feedback"| C
        D -.->|"Result: /action/result"| C
        C -.->|"Cancel: /action/cancel"| D
    end
``` -->

## Flow Diagram

![Architecture Diagram](/img/ch3-flow.svg)

![Architecture Diagram](/img/ch3-flow2.svg)
<!-- 
<div
  style={{
    textAlign: "center",
    border: "1px solid #ccc",
    padding: "10px",
    marginBottom: "20px"
  }}
>
  <img
    src="/img/ch3-flow.svg"
    alt="Flow Diagram 1"
    style={{ maxWidth: "400px", width: "80%" }}
  />
</div>

<div
  style={{
    textAlign: "center",
    border: "1px solid #ccc",
    padding: "10px",
    marginBottom: "20px"
  }}
>
  <img
    src="/img/ch3-flow2.svg"
    alt="Flow Diagram 2"
    style={{ maxWidth: "400px", width: "80%" }}
  />
</div> -->


<!-- ```mermaid
sequenceDiagram
    participant C as Service Client
    participant S as Service Server

    C->>S: Request
    S->>S: Process Request
    S->>C: Response
``` -->

<!-- ```mermaid
sequenceDiagram
    participant AC as Action Client
    participant AS as Action Server

    AC->>AS: Send Goal
    AS->>AS: Execute Goal
    AS->>AC: Feedback (repeated)
    AS->>AC: Result
``` -->

## Code Example: Service Server and Client

Here's an example of a ROS2 service server:

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response


def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

And the corresponding service client:

```python
import sys
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class MinimalClient(Node):

    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()


def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClient()
    response = minimal_client.send_request(int(sys.argv[1]), int(sys.argv[2]))
    minimal_client.get_logger().info(
        'Result of add_two_ints: %d' % response.sum)
    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Code Example: Action Server and Client

Action server example:

```python
import time
import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from custom_interfaces.action import Fibonacci


class FibonacciActionServer(Node):

    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            execute_callback=self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

    def destroy(self):
        self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        self.get_logger().info('Received goal request')
        return rclpy.action.server.GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Feedback: {feedback_msg.sequence}')
            time.sleep(1)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info(f'Result: {result.sequence}')
        return result


def main(args=None):
    rclpy.init(args=args)
    fibonacci_action_server = FibonacciActionServer()

    executor = MultiThreadedExecutor()
    rclpy.spin(fibonacci_action_server, executor=executor)

    fibonacci_action_server.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Custom Service Example

To create a custom service type, create a `.srv` file in your package's `srv` directory:

```
# CalculateDistance.srv
float64 x1
float64 y1
float64 x2
float64 y2
---
float64 distance
string error_message
```

## Step-by-Step Practical Tutorial

### Creating a Service Server and Client

1. **Create a new ROS2 package**:
   ```bash
   cd ~/ros2_ws/src
   ros2 pkg create --build-type ament_python my_robot_services
   ```

2. **Create the service server** (`my_robot_services/my_robot_services/service_server.py`):
   ```python
   # Use the service server code example above
   ```

3. **Create the service client** (`my_robot_services/my_robot_services/service_client.py`):
   ```python
   # Use the service client code example above
   ```

4. **Make the scripts executable**:
   ```bash
   chmod +x my_robot_services/my_robot_services/service_server.py
   chmod +x my_robot_services/my_robot_services/service_client.py
   ```

5. **Update the setup.py file** to include entry points:
   ```python
   entry_points={
       'console_scripts': [
           'service_server = my_robot_services.service_server:main',
           'service_client = my_robot_services.service_client:main',
       ],
   },
   ```

6. **Build the package**:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select my_robot_services
   ```

7. **Source the workspace**:
   ```bash
   source install/setup.bash
   ```

8. **Run the service server in one terminal**:
   ```bash
   ros2 run my_robot_services service_server
   ```

9. **In another terminal, run the service client**:
   ```bash
   ros2 run my_robot_services service_client 2 3
   ```

## Summary

This chapter explored ROS2 services and actions, which provide synchronous and goal-based communication patterns respectively. Services are ideal for request-response interactions, while actions are designed for long-running tasks that require feedback and cancellation capabilities.

Understanding when to use topics, services, or actions is crucial for designing effective robotic systems. Topics for streaming data, services for immediate requests, and actions for complex operations with progress tracking.

## Mini-Quiz

1. What is the main difference between ROS2 services and topics?
   - A) Services are faster than topics
   - B) Services provide synchronous request-response while topics are asynchronous
   - C) Services use different message types
   - D) There is no difference

2. Which communication pattern is best for a long-running navigation task that needs to provide feedback?
   - A) Topic
   - B) Service
   - C) Action
   - D) Parameter

3. What can a ROS2 action provide that a service cannot?
   - A) Request and response
   - B) Feedback during execution
   - C) Error handling
   - D) Authentication

4. In a service communication, what happens when the client sends a request?
   - A) The client continues execution immediately
   - B) The client waits for the response (blocking)
   - C) The server ignores the request
   - D) The request is queued indefinitely

5. Which of the following can be canceled once started?
   - A) Service call
   - B) Topic publishing
   - C) Action goal
   - D) Parameter update

**Answers**: 1-B, 2-C, 3-B, 4-B, 5-C