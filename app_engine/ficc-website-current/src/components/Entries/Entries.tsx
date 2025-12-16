import { FC } from 'react';
import Section from '@components/Section';
import Region from '@components/Region';
import Aside from '@components/Aside';

import type { EntriesProps as Props } from './types';


const Entries: FC<Props> = ({ entries }) => {
  if (!entries?.length) {
    console.warn(
      `Entries did not render because prop 'entries' is ${entries}`,
    );

    return null;
  }

  return (
    <>
      {entries.map(item => {
        switch(item.__typename) {
          case 'ComponentRegion': {
            return (
              <Region
                key={item.sys.id}
                {...item}
              />
            );
          }
          case 'ComponentSection': {
            return (
              <Section
                key={item.sys.id}
                {...item}
                parent
              />
            );
          }
          case 'ComponentCallout': {
            return (
              <Aside
                key={item.sys.id}
                {...item}
              />
            );
          }
          default: {
            return null;
          }
        }
      })}
    </>
  );
};


export default Entries;
