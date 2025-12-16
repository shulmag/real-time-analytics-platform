export const Asset = `#graphql
  ... on Asset {
    __typename
    sys {
      id
    }
    url
    fileName
    description
  }
`;
