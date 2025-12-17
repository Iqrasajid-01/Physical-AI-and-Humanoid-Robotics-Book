<<<<<<< HEAD
import React, { useState } from 'react';
import Link from '@docusaurus/Link';
import styles from './table-of-contents.module.css';

interface CardProps {
  title: string;
  link?: string;
  children?: React.ReactNode;
}

const Card: React.FC<CardProps> = ({ title, link, children }) => {
  const [expanded, setExpanded] = useState(false);
  const toggleExpand = () => setExpanded(!expanded);

  return (
    <div className={styles.card} onClick={children ? toggleExpand : undefined}>
      {link ? (
        <Link to={link} className={styles.cardLink}>
          {title}
        </Link>
      ) : (
        <span className={styles.cardTitle}>{title}</span>
      )}
      {expanded && children && <div className={styles.nested}>{children}</div>}
    </div>
  );
};

export default function TableOfContents() {
  return (
    <div className={styles.container}>
      <Link to="/" className={styles.backButton}>
        ← Back
      </Link>
      <h1 className={styles.heading}>Table of Contents</h1>
      <div className={styles.cardsWrapper}>
        <Card title="Modules">
          <Card title="Module 1" link="/docs/module1_ros2/01-introduction-to-ros2" />
          <Card title="Module 2" link="/docs/module2_simulation/07-introduction-to-gazebo-simulation" />
          <Card title="Module 3" link="/docs/module3_isaac/12-introduction-to-nvidia-isaac" />
          <Card title="Module 4" link="/docs/module4_vla/18-vision-language-action-models" />
        </Card>
        <Card title="Labs and Projects" link="/docs/labs_and_projects/23-robotics-labs" />
        <Card title="Capstone" link="/docs/capstone/25-capstone-design" />
        <Card title="Assessment" link="/docs/assessment/27-assessment-and-validation" />
        <Card title="Appendices" link="/docs/appendices/28-appendices" />
      </div>
    </div>
  );
}
=======
import React, { useState } from 'react';
import Link from '@docusaurus/Link';
import styles from './table-of-contents.module.css';

interface CardProps {
  title: string;
  link?: string;
  children?: React.ReactNode;
}

const Card: React.FC<CardProps> = ({ title, link, children }) => {
  const [expanded, setExpanded] = useState(false);
  const toggleExpand = () => setExpanded(!expanded);

  return (
    <div className={styles.card} onClick={children ? toggleExpand : undefined}>
      {link ? (
        <Link to={link} className={styles.cardLink}>
          {title}
        </Link>
      ) : (
        <span className={styles.cardTitle}>{title}</span>
      )}
      {expanded && children && <div className={styles.nested}>{children}</div>}
    </div>
  );
};

export default function TableOfContents() {
  return (
    <div className={styles.container}>
      <Link to="/" className={styles.backButton}>
        ← Back
      </Link>
      <h1 className={styles.heading}>Table of Contents</h1>
      <div className={styles.cardsWrapper}>
        <Card title="Modules">
          <Card title="Module 1" link="/docs/module1_ros2/01-introduction-to-ros2" />
          <Card title="Module 2" link="/docs/module2_simulation/07-introduction-to-gazebo-simulation" />
          <Card title="Module 3" link="/docs/module3_isaac/12-introduction-to-nvidia-isaac" />
          <Card title="Module 4" link="/docs/module4_vla/18-vision-language-action-models" />
        </Card>
        <Card title="Labs and Projects" link="/docs/labs_and_projects/23-robotics-labs" />
        <Card title="Capstone" link="/docs/capstone/25-capstone-design" />
        <Card title="Assessment" link="/docs/assessment/27-assessment-and-validation" />
        <Card title="Appendices" link="/docs/appendices/28-appendices" />
      </div>
    </div>
  );
}
>>>>>>> 8c6d10ec56bb3d36e488625ab843181aaa3fa81a
