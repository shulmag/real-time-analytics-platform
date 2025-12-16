import { FC } from 'react';
import State from '@providers/StateProvider';
import Head from './Head';
import Hero, { HeroProps } from './Hero';
import Nav from './Nav';
import Footer from './Footer';
import { isElement } from '#utils';


interface Props {
  readonly slug: string;
  readonly hero: HeroProps | JSX.Element;
  readonly head: {
    title: string;
    description: string;
  };
  readonly links: Array<string>;
  readonly children: JSX.Element;
}

const Layout: FC<Props> = ({
  slug,
  hero,
  head,
  links,
  children,
}) => (
  <State slug={slug}>
    <Head {...head} />
    {isElement(hero) ? hero : <Hero {...hero} />}
    <Nav links={links} />
    <main>{children}</main>
    <Footer links={links} />
  </State>
);


export default Layout;
