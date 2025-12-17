# Data Model: Textbook on Physical AI & Humanoid Robotics

## Textbook Chapter
- **name**: string - Chapter identifier (e.g., "01-introduction-to-ros2")
- **title**: string - Human-readable chapter title
- **module**: string - Module category (ROS2, Simulation, Isaac, VLA, Labs, Capstone, Appendices)
- **number**: integer - Sequential chapter number (01-28)
- **objectives**: array of strings - Learning objectives for the chapter
- **core_concepts**: array of strings - Key concepts covered
- **diagrams**: array of objects - Diagrams included in the chapter
  - type: string (architecture, flow, annotated_image, block)
  - path: string - Path to diagram asset
  - caption: string - Description of the diagram
- **code_examples**: array of objects - Code snippets in the chapter
  - language: string (python, xml, yaml, etc.)
  - content: string - Code content
  - purpose: string - What the code demonstrates
  - type: string (minimal, full_working)
- **tutorial_steps**: array of strings - Step-by-step tutorial instructions
- **summary**: string - Chapter summary
- **quiz_questions**: array of objects - Mini-quiz questions
  - question: string - The quiz question
  - options: array of strings - Multiple choice options
  - correct_answer: string - Index or text of correct answer
  - explanation: string - Why this is the correct answer

## Learning Module
- **name**: string - Module identifier (e.g., "module1_ros2")
- **title**: string - Module title
- **chapters**: array of strings - Chapter names in this module
- **description**: string - Overview of the module content
- **prerequisites**: array of strings - Knowledge needed before this module
- **learning_outcomes**: array of strings - What students will learn

## Visual Asset
- **id**: string - Unique identifier for the asset
- **filename**: string - Name of the file in assets directory
- **type**: string (diagram, screenshot, flowchart, annotated_image)
- **title**: string - Brief description
- **usage**: array of strings - Which chapters use this asset
- **alt_text**: string - Accessibility text for the image
- **caption**: string - Descriptive caption

## Code Example
- **id**: string - Unique identifier
- **title**: string - Brief description of the example
- **language**: string (python, xml, yaml, etc.)
- **content**: string - Full code content
- **chapter**: string - Chapter this belongs to
- **type**: string (minimal, full_working)
- **purpose**: string - What concept this demonstrates
- **dependencies**: array of strings - Other code files or libraries needed
- **expected_output**: string - What the code should produce when run
- **tested**: boolean - Whether the code has been verified to work

## Assessment Question
- **id**: string - Unique identifier
- **chapter**: string - Chapter this question belongs to
- **question_text**: string - The actual question
- **question_type**: string (multiple_choice, true_false, short_answer)
- **options**: array of strings - For multiple choice questions
- **correct_answer**: string - The correct answer
- **explanation**: string - Why this is correct
- **difficulty**: string (beginner, intermediate, advanced)