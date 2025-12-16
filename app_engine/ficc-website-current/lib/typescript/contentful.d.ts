/**
 * Internal
 */

export interface ContentfulBase<T = string> {
  readonly __typename: T;
  readonly sys: R<{
    id: string;
  }>;
}

export interface ContentfulAsset extends ContentfulBase {
  readonly url: string;
  readonly fileName: string;
  readonly description: string;
}


/**
 * Entries
 */

export interface ComponentImage extends ContentfulBase<'ComponentImage'> {
  readonly title: _string;
  readonly desktop: Maybe<ContentfulAsset>;
  readonly mobile: Maybe<ContentfulAsset>;
}

export interface ComponentButton extends ContentfulBase<'ComponentButton'> {
  readonly text: string;
  readonly external: _string;
  readonly internal: Maybe<R<{
    slug: _string;
  }>>;
}

export interface ComponentHero extends ContentfulBase<'ComponentHero'> {
  readonly heading: string;
  readonly copy: _string;
  readonly image: Maybe<ComponentImage>;
  readonly cta: Maybe<ComponentButton>;
}

export interface ComponentSection extends ContentfulBase<'ComponentSection'> {
  readonly heading: _string;
  readonly subHeading: _string;
  readonly copy: _string;
  readonly icon: Maybe<boolean>;
  readonly contactForm: Maybe<boolean>;
}

export interface ComponentRegion extends ContentfulBase<'ComponentRegion'> {
  readonly heading: _string;
  readonly sections: R<{
    items: Maybe<RA<ComponentSection>>;
  }>;
}

export interface ComponentCallout extends ContentfulBase<'ComponentCallout'> {
  readonly heading: _string;
  readonly copy: _string;
}

export interface ComponentNav extends ContentfulBase<'ComponentNav'> {
  readonly links: Array<string>;
}

export type ContentUnion =
  | ComponentRegion
  | ComponentSection
  | ComponentCallout;


/**
 * Pages
 */

export interface ContentfulPage extends ContentfulBase {
  readonly title: _string;
  readonly description: _string;
  readonly slug: _string;
  readonly content: R<{
    items: Maybe<Readonly<[ComponentHero, ContentUnion]>>;
  }>;
}


/**
 * Main Export
 */

export interface Contentful {
  Base: ContentfulBase;
  Asset: ContentfulAsset;
  Image: ComponentImage;
  Button: ComponentButton;
  Hero: ComponentHero;
  Section: ComponentSection;
  Region: ComponentRegion;
  Callout: ComponentCallout;
  Page: ContentfulPage;
  Nav: ComponentNav;
  Data: { page: ContentfulPage, nav: ComponentNav };
  Entry: ComponentHero | ContentUnion;
}
