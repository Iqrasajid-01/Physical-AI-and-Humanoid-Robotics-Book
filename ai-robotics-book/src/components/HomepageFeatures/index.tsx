import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'About This Book',
    Svg: require('@site/static/img/undraw_reading-a-book_4cap.svg').default,
    description: (
      <>
        This textbook covers Physical AI and Humanoid Robotics with ROS2, Gazebo, and Isaac. 
        It’s perfect for both beginners and experts, offering hands-on tutorials, simulations, and real-world AI applications.
      </>
    ),
  },
  {
    title: 'What You Will Learn',
    Svg: require('@site/static/img/undraw_online-learning_tgmv.svg').default,
    description: (
      <>
        
        Setting up ROS2 for robotics<br />
      Simulating humanoid robots in Gazebo<br />
      AI for autonomous control<br />
      Humanoid navigation & path planning<br />
      Integrating voice commands with AI & NLP
      
      </>
    ),
  },
  {
    title: 'Key Features',
    Svg: require('@site/static/img/undraw_visual-explanation_vd4l.svg').default,
    description: (
      <>
        Tutorials with code<br />
      Gazebo & Isaac simulations<br />
      AI & machine learning<br />
      Humanoid robot control
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
