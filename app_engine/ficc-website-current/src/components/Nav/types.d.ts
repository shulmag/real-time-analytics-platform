import type { ComponentPropsWithoutRef } from 'react';

interface NavBase extends ComponentPropsWithoutRef<'nav'> {
  readonly links: Array<string>;
  children?: never;
}

export type NavProps = Partial<NavBase>;
