import { Section } from './Section';


export const Region = `#graphql
  ... on ComponentRegion {
    __typename
    sys {
      id
    }
    heading
    sections: sectionsCollection(limit: 10) {
      items {
        ${Section}
      }
    }
  }
`;
