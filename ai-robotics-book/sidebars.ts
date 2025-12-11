import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 * - create an ordered group of docs
 * - render a sidebar for each doc of that group
 * - provide next/previous navigation
 *
 * The sidebars can be generated from the filesystem, or explicitly defined here.
 *
 * Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Module 1 - ROS2',
      items: [
        'module1_ros2/introduction-to-ros2',
        'module1_ros2/ros2-nodes-and-topics',
        'module1_ros2/services-and-actions',
        'module1_ros2/ros2-packages-and-launch-files',
        'module1_ros2/ros2-parameters-and-configuration',
        'module1_ros2/ros2-best-practices',
      ],
    },
    {
      type: 'category',
      label: 'Module 2 - Simulation',
      items: [
        'module2_simulation/introduction-to-gazebo-simulation',
        'module2_simulation/robot-modeling-with-urdf',
        'module2_simulation/sensors-and-physics-in-simulation',
        'module2_simulation/unity-isaac-sim-basics',
        'module2_simulation/simulation-integration',
      ],
    },
    {
      type: 'category',
      label: 'Module 3 - Isaac',
      items: [
        'module3_isaac/introduction-to-nvidia-isaac',
        'module3_isaac/isaac-perception-systems',
        'module3_isaac/isaac-navigation-nav2',
        'module3_isaac/isaac-ros-integration',
        'module3_isaac/isaac-vslam-implementation',
        'module3_isaac/isaac-best-practices',
      ],
    },
    {
      type: 'category',
      label: 'Module 4 - VLA',
      items: [
        'module4_vla/vision-language-action-models',
        'module4_vla/vla-planning-algorithms',
        'module4_vla/integration-with-robotics',
        'module4_vla/whisper-integration-for-voice-commands',
        'module4_vla/advanced-vla-applications',
      ],
    },
    {
      type: 'category',
      label: 'Labs and Projects',
      items: [
        'labs_and_projects/robotics-labs',
        'labs_and_projects/project-implementation',
      ],
    },
    {
      type: 'category',
      label: 'Assessment',
      items: [
        'assessment/assessment-and-validation',
      ],
    },
    {
      type: 'category',
      label: 'Capstone',
      items: [
        'capstone/capstone-design',
        'capstone/capstone-implementation',
      ],
    },
    {
      type: 'category',
      label: 'Appendices',
      items: [
        'appendices/appendices',
      ],
    },
  ],

  // But you can create a sidebar manually
  /*
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Tutorial',
      items: ['intro', 'hello'],
    },
  ],
   */
};

export default sidebars;