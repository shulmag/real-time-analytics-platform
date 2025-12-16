import Layout from '@components/Layout';
import Entries from '@components/Entries';
import { request } from '#utils';
import { Entry } from '#constants';
import { Page } from '#graphql';

import type { NextPage, GetStaticProps } from 'next';
import type { Contentful } from '_contentful';


const production: boolean = process.env.NODE_ENV === 'production';

interface Props {
  readonly page: Contentful['Page'];
  readonly nav: Contentful['Nav'];
}

const Home: NextPage<Props> = ({ page, nav }) => {
  if (!page) {
    console.warn(
      `Index did not render because prop 'page' is ${page}`,
    );

    return null;
  }

  const [hero, ...content] = page.content.items || [];

  return (
    <Layout
      slug={page.slug || '/'}
      hero={hero || {}}
      head={{
        title: page.title || '',
        description: page.description || '',
      }}
      links={nav.links}
    >
      <Entries entries={content} />
    </Layout>
  );
};

const getStaticProps: GetStaticProps = async () => {
  const { page, nav } = await request<Contentful['Data']>(`#graphql
    {
      page(
        id: "${Entry.HOME}"
        preview: ${!production}
      ) {
        ${Page}
      }
      nav: componentNav(
        id: "${Entry.NAV}"
        preview: ${!production}
      ) {
        links
      }
    }
  `);

  return {
    props: {
      page,
      nav,
    },
    revalidate: 120,
  };
};


export default Home;
export { getStaticProps };
