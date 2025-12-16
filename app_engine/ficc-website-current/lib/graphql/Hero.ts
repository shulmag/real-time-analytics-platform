import { Image } from './Image';
import { Button } from './Button';


export const Hero = `#graphql
  ... on ComponentHero {
    __typename
    sys {
      id
    }
    heading
    copy
    image {
      ${Image}
    }
    cta {
      ${Button}
    }
  }
`;
