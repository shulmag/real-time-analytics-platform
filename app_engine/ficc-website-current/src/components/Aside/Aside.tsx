import { FC } from 'react';
import Markdown, { MarkdownToJSX } from 'markdown-to-jsx';
import styles from './Aside.module.scss';

import type { AsideProps as Props } from './types';


const options: MarkdownToJSX.Options = { forceInline: true };

const Aside: FC<Props> = ({ heading, copy }) => (
  <aside
    className={styles.root}
    aria-label={heading || undefined}
  >
    <div>
      {heading && <h3 className="outline">{heading}</h3>}
      {copy && <p><Markdown options={options}>{copy}</Markdown></p>}
    </div>
  </aside>
);


export default Aside;
