import type { ComponentPropsWithoutRef } from 'react';
import type { Contentful } from '_contentful';


type RegionBase = Contentful['Region'] &
  ComponentPropsWithoutRef<'section'> &
  { children?: never };

export type RegionProps = Partial<RegionBase>;
