import React from 'react';
import styles from './CustomFooter.module.css';
import { FaGithub, FaFacebook, FaLinkedin, FaInstagram, FaYoutube } from 'react-icons/fa';

export default function CustomFooter() {
  return (
    <footer className={styles.footer}>
      <div className={styles.links}>
        <a href="https://www.youtube.com/watch?v=YOUR_VIDEO_LINK" target="_blank" rel="noopener noreferrer">
          <FaYoutube size={24} />
        </a>
        <a href="https://github.com/YOUR_GITHUB_USERNAME" target="_blank" rel="noopener noreferrer">
          <FaGithub size={24} />
        </a>
        <a href="https://www.facebook.com/YOUR_FACEBOOK_PROFILE" target="_blank" rel="noopener noreferrer">
          <FaFacebook size={24} />
        </a>
        <a href="https://www.linkedin.com/in/YOUR_LINKEDIN_PROFILE" target="_blank" rel="noopener noreferrer">
          <FaLinkedin size={24} />
        </a>
        <a href="https://www.instagram.com/YOUR_INSTAGRAM_PROFILE" target="_blank" rel="noopener noreferrer">
          <FaInstagram size={24} />
        </a>
      </div>
      <p>© {new Date().getFullYear()} Created by Iqra Sajid</p>
    </footer>
  );
}
