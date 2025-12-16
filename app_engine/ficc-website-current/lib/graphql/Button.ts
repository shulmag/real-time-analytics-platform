export const Button = `#graphql
  ... on ComponentButton {
    __typename
    sys {
      id
    }
    text
    external
    internal {
      slug
    }
  }
`;
