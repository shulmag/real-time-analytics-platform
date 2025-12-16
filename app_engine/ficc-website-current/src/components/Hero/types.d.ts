import type { ComponentPropsWithoutRef } from 'react';
import type { Contentful } from '_contentful';


type HeroBase = Contentful['Hero']
  & ComponentPropsWithoutRef<'header'>
  & { children?: never };

export type HeroProps = Partial<HeroBase>;
