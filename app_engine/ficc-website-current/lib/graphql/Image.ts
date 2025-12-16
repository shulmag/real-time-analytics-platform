import { Asset } from './Asset';


export const Image = `#graphql
  ... on ComponentImage {
    __typename
    sys {
      id
    }
    title
    desktop {
      ${Asset}
    }
    mobile {
      url
    }
  }
`;
