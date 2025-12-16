import type { Contentful } from '_contentful';

interface EntriesBase {
  readonly entries: ReadonlyArray<Contentful['Entry']>;
}

export type EntriesProps = Partial<EntriesBase>;
