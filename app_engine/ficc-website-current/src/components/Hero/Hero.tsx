import { FC } from 'react';
import Markdown, { MarkdownToJSX } from 'markdown-to-jsx';
import { state } from '@providers/StateProvider';
import { createSpan } from '#hooks';
import { cls } from '#utils';
import styles from './Hero.module.scss';

import type { HeroProps as Props } from './types';


const options: MarkdownToJSX.Options = {
  wrapper: 'h1',
  forceWrapper: true,
};

const Hero: FC<Props> = ({ heading, copy }) => {
  const { slug } = state();
  const title = createSpan(heading);

  return (
    <header className={cls(styles.root, slug !== '/' && styles.sibling)} >
      <div data-grid>
        <div className={styles.content}>
          <Markdown options={options}>{title}</Markdown>
          {copy && <p>{copy}</p>}
        </div>
        {slug === '/' && (
          <picture className={styles.image}>
            <img
              src="/img/ficc-hero.svg"
              srcSet="/img/ficc-hero.svg"
              alt=""
            />
          </picture>
        )}
      </div>
    </header>
  );
};


export default Hero;
