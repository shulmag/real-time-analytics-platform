import { createSlug } from '#hooks';
import styles from './Footer.module.scss';

import type { FC } from 'react';
import type { FooterProps as Props } from './types';

const Footer: FC<Props> = ({ links }) => {
  const year = new Date().getFullYear();

  return (
    <footer className={styles.root}>
      <div data-wrap>
        <nav>
          <a
            className="Logo"
            href="#"
          >
            ficc<span>.ai</span>
          </a>
          {links?.length && links.map(item => (
            <a
              href={`/#${createSlug(item)}`}
              key={createSlug(item)}
            >
              {item}
            </a>
          ))}
        </nav>
        <section>
          <p><small><abbr title="Copyright">©</abbr> {year} ficc.ai</small></p>
          <a href="/privacy-and-terms">Privacy &amp; Terms</a>
        </section>
      </div>
    </footer>
  );
};


export default Footer;
