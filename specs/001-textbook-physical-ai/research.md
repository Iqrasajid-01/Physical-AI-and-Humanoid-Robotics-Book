# Research: Textbook on Physical AI & Humanoid Robotics

## Decision: Technology Stack and Platform
**Rationale**: The textbook will be built using Docusaurus, a modern static site generator, as evidenced by the existing project structure in `/ai-robotics-book/`. This provides an excellent platform for documentation-based content with support for Markdown, code snippets, and visual assets.

**Alternatives considered**:
- Custom static site generator
- WordPress-based solution
- GitBook

## Decision: Module Sequence & Chapter Order
**Rationale**: Following the functional requirement FR-001, the sequence will be:
1. ROS2 Module (chapters 01-06): Foundation concepts
2. Simulation Module (chapters 07-11): Building on ROS2 knowledge
3. Isaac Module (chapters 12-17): Advanced perception and control
4. VLA Module (chapters 18-22): AI integration
5. Labs & Projects (chapters 23-24): Integration exercises
6. Capstone & Assessment (chapters 25-26): Complete system implementation
7. Appendices (chapters 27-28): Reference materials

This follows a progressive learning approach from simple to complex concepts.

## Decision: Diagram Types and Placement
**Rationale**: Each chapter will include 2-3 diagrams as specified in FR-003:
- 1 architecture or system diagram per chapter to show component relationships
- 1 flow diagram for processes or algorithms
- 1-2 annotated images for visual learning support

Diagrams will be placed after the core concepts section but before the practical tutorial to help visualize the concepts before implementation.

## Decision: Code Example Depth
**Rationale**: Following FR-010, each chapter will include both:
- Minimal examples: Small, focused code snippets that demonstrate a single concept
- Full working examples: Complete implementations that students can run and modify

This provides both quick understanding and practical application opportunities.

## Decision: Capstone Implementation
**Rationale**: The capstone chapter (chapter 25-26) will demonstrate a complete autonomous humanoid system integrating all previous modules. It will include a full end-to-end system diagram as specified in FR-007, showing how ROS2, simulation, Isaac perception, and VLA planning work together.

## Decision: Pedagogical Approach
**Rationale**: Following the constitution's "Progressive Learning (NON-NEGOTIABLE)" principle and FR-006, each chapter will:
- Start with simple concepts and gradually introduce complexity
- Use example-led explanations with hands-on exercises
- Include practical tutorials with step-by-step instructions
- Build on previous knowledge while introducing new concepts

## Decision: Docusaurus Configuration
**Rationale**: The sidebar.js will be configured to reflect the module structure with:
- Main modules as top-level categories
- Individual chapters organized under appropriate modules
- Proper navigation flow from basic to advanced topics
- Search functionality for easy content discovery

This ensures the textbook is fully navigable as specified in FR-008.