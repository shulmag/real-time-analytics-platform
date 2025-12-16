import type { ComponentPropsWithoutRef } from 'react';
import type { Contentful } from '_contentful';


type SectionBase = Contentful['Section'] &
  ComponentPropsWithoutRef<'section'> &
  {
    parent?: boolean;
    children?: never;
};

export type FormDataType = {
  [x: string]: string | number | boolean;
  'form-name': string;
  name: string;
  email: string;
};

export type SectionProps = Partial<SectionBase>;
