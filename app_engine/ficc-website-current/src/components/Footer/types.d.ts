import type { ComponentPropsWithoutRef } from 'react';

interface FooterBase extends ComponentPropsWithoutRef<'footer'> {
  readonly links: Array<string>;
  children?: never;
}

export type FooterProps = Partial<FooterBase>;
