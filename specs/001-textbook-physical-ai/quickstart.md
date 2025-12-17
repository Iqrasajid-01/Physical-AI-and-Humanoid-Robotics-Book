# Quickstart Guide: Textbook on Physical AI & Humanoid Robotics

## Prerequisites

Before starting with the textbook, ensure you have the following:

### System Requirements
- Operating System: Ubuntu 20.04/22.04 LTS, macOS 10.15+, or Windows 10/11 with WSL2
- RAM: 8GB minimum (16GB recommended)
- Storage: 10GB free space for all dependencies
- Internet connection for downloading packages and dependencies

### Software Dependencies
- Node.js v18 or higher
- npm or yarn package manager
- Git for version control
- Python 3.8 or higher (for ROS2 Humble/Iron)
- Docker (optional, for simulation environments)

## Setup Environment

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/ai-robotics-book.git
cd ai-robotics-book
```

### 2. Install Dependencies
```bash
npm install
# or if using yarn
yarn install
```

### 3. Start the Docusaurus Development Server
```bash
npm run start
# or
yarn start
```

The textbook will be available at `http://localhost:3000`

## Textbook Structure

The textbook is organized into 4 main modules:

### Module 1: ROS2 (Chapters 01-06)
- Chapter 01: Introduction to ROS2
- Chapter 02: ROS2 Nodes and Topics
- Chapter 03: Services and Actions
- Chapter 04: ROS2 Packages and Launch Files
- Chapter 05: ROS2 Parameters and Configuration
- Chapter 06: ROS2 Best Practices

### Module 2: Simulation (Chapters 07-11)
- Chapter 07: Introduction to Gazebo Simulation
- Chapter 08: Robot Modeling with URDF
- Chapter 09: Sensors and Physics in Simulation
- Chapter 10: Unity Isaac Sim Basics
- Chapter 11: Simulation Integration

### Module 3: Isaac (Chapters 12-17)
- Chapter 12: Introduction to NVIDIA Isaac
- Chapter 13: Isaac Perception Systems
- Chapter 14: Isaac Navigation (Nav2)
- Chapter 15: Isaac ROS Integration
- Chapter 16: Isaac VSLAM Implementation
- Chapter 17: Isaac Best Practices

### Module 4: VLA (Chapters 18-22)
- Chapter 18: Vision-Language-Action Models
- Chapter 19: VLA Planning Algorithms
- Chapter 20: Integration with Robotics
- Chapter 21: Whisper Integration for Voice Commands
- Chapter 22: Advanced VLA Applications

### Module 5: Labs & Projects (Chapters 23-24)
- Chapter 23: Robotics Labs
- Chapter 24: Project Implementation

### Module 6: Capstone & Assessment (Chapters 25-26)
- Chapter 25: End-to-End System Design
- Chapter 26: Autonomous Humanoid Operation

### Module 7: Appendices (Chapters 27-28)
- Chapter 27: Reference Materials
- Chapter 28: Troubleshooting Guide

## How to Use This Textbook

1. **Start with Module 1**: Each module builds on previous knowledge
2. **Follow the Practical Tutorials**: Execute code examples as you read
3. **Complete Mini-Quizzes**: Test your understanding at the end of each chapter
4. **Use Visual Aids**: Study diagrams and annotated images to understand concepts
5. **Progress at Your Own Pace**: Each chapter is designed to be self-contained while building on previous concepts

## Running Code Examples

Code examples are provided in multiple formats:

- **Minimal Examples**: Small snippets to demonstrate specific concepts
- **Full Working Examples**: Complete implementations that you can run

To run Python ROS2 examples:
```bash
cd ai-robotics-book/examples/chapter-01
python3 example_node.py
```

## Getting Help

- Check the Troubleshooting Guide in Chapter 28
- Refer to the reference materials in Chapter 27
- Use the search functionality to find specific topics
- Join our community forums for additional support