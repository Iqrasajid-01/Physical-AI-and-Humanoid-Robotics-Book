<!-- Sync Impact Report:
Version change: 1.0.0 → 1.0.0 (initial creation)
Modified principles: N/A (new creation)
Added sections: All sections added as this is initial creation
Removed sections: N/A
Templates requiring updates: ⚠ pending - .specify/templates/plan-template.md, .specify/templates/spec-template.md, .specify/templates/tasks-template.md
Follow-up TODOs: None
-->
# AI-Native Textbook — Physical AI & Humanoid Robotics Constitution

## Core Principles

### I. Educational Excellence
Content must be pedagogically sound with clear learning objectives, hands-on tutorials, and practical examples. Every chapter must include objectives, core concepts, summary, and mini-quiz (3-5 questions) to ensure student comprehension and retention.

### II. Technical Accuracy
All technical content must be accurate and current with ROS2 Humble/Iron, Gazebo, Unity Isaac Sim, Nav2, VLA, and Whisper technologies. Code examples must be tested and verified to work in the target environments before inclusion in the textbook.

### III. Progressive Learning (NON-NEGOTIABLE)
Content must follow a progression from simple to complex concepts with example-led, hands-on approach. Each chapter builds upon previous knowledge while introducing new concepts in an accessible manner for diverse technical backgrounds.

### IV. Practical Application
Every concept must include practical hands-on tutorials with real-world examples. Students must be able to implement what they learn through step-by-step instructions with expected outcomes and troubleshooting guidance.

### V. Visual Learning Support
All chapters must include appropriate diagrams (architecture, flow, annotated images) and visual aids (ROS/simulation/block diagrams) to support different learning styles and complex concept comprehension.

### VI. Code Quality Standards
All code examples must follow best practices for Python ROS2 (rclpy), URDF/XML, YAML, launch files, Isaac Sim, VLA planning, and Whisper integration. Code must be well-documented with clear comments and follow consistent formatting standards.

## Content Standards

### Chapter Requirements
Each chapter must include: Objectives, introduction, core concepts, practical tutorial with hands-on steps, summary, and mini-quiz (3-5 questions). Chapters must contain diagrams, code examples, and visual assets referenced via `/assets/...` path structure.

### Documentation Format
All content must be in Markdown format with fenced code blocks. Visual assets must be properly referenced and included in the `/assets` directory. Content must follow the specified folder structure with sequential numbering (01-28) and module organization.

### Success Criteria
Students must be able to: Build ROS2 nodes/packages, Simulate robots (Gazebo/Unity), Run Isaac ROS VSLAM, implement Nav2 navigation, execute VLA planning, and complete the capstone project as specified in the textbook.

## Development Workflow

### Content Creation Process
All chapters follow the spec-driven approach: Requirements defined → Content drafted → Technical verification → Peer review → Integration into textbook structure. Each chapter must pass technical validation before being marked complete.

### Quality Assurance
All content undergoes: Technical accuracy verification, Educational effectiveness review, Code example testing, Visual asset validation, and Cross-reference consistency check. Each chapter must meet pedagogical and technical standards before approval.

### Version Control
All content changes follow Git workflow with descriptive commit messages. Branches follow feature-based naming convention. All changes must pass review process before merging to main textbook content.

## Governance

This constitution governs all development of the AI-Native Textbook on Physical AI & Humanoid Robotics. All contributors must adhere to these principles and standards. Amendments require project lead approval and must maintain educational excellence and technical accuracy standards. All PRs and reviews must verify compliance with these principles. Complexity must be justified with clear educational value.

**Version**: 1.0.0 | **Ratified**: 2025-12-09 | **Last Amended**: 2025-12-09