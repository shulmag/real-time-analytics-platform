import { FC } from 'react';
import Markdown, { MarkdownToJSX } from 'markdown-to-jsx';
import Section from '@components/Section';
import { createSlug, createSpan } from '#hooks';
import { cls, isOdd } from '#utils';
import styles from './Region.module.scss';

import type { RegionProps as Props } from './types';


const options: MarkdownToJSX.Options = {
  wrapper: 'h2',
  forceWrapper: true,
};

const Region: FC<Props> = ({ heading, sections }) => {
  const _heading = heading || '';
  const outline: boolean = /company/i.test(_heading);
  const team: boolean = /team|leader/i.test(_heading);
  const advisors: boolean = /advisors|senior/i.test(_heading);
  const title: string = createSpan(_heading);

  return (
    <section
      id={createSlug(_heading)}
      className={cls(
        styles.root,
        team && styles.team,
        outline && styles.company,
      )}
    >
      <div
        data-grid={team ? true : undefined}
        data-wrap={advisors ? true : undefined}
      >
        <Markdown
          options={options}
          className={cls(outline && 'outline')}
        >
          {title}
        </Markdown>
        <div data-sub-grid>
          {sections?.items?.map(section => (
            <Section
              key={section.sys.id}
              {...section}
            />
          ))}
          {team && isOdd(sections?.items?.length) && (
            <div className="filler" />
          )}
        </div>
      </div>
    </section>
  );
};


export default Region;
