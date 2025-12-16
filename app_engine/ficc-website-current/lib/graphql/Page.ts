import { Hero } from './Hero';
import { Section } from './Section';
import { Region } from './Region';
import { Callout } from './Callout';


export const Page = `#graphql
  slug
  title
  description
  content: contentCollection(limit: 10) {
    items {
      ${Hero}
      ${Section}
      ${Region}
      ${Callout}
    }
  }
`;
