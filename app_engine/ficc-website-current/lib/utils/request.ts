import { GraphQLClient, gql } from 'graphql-request';


const { SPACE } = process.env;
const { TOKEN } = process.env;

export const request = async <Q>(query: string): Promise<Q> => {
  const client = new GraphQLClient(
    `https://graphql.contentful.com/content/v1/spaces/6n4d1tc2ev18`,
    {
      headers: {
        'Content-Type': 'application/json',
        'authorization': `Bearer DqQLJsgMqXofsgsRi7Vz0Z4aE_bAN2JtZW0Xwj7An3s`,
      },
    },
  );

  const data = await client.request(gql`${query}`);

  return data;
};
