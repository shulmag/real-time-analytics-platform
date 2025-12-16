import NextHead from 'next/head';
import useConfig from 'next/config';

import type { FC } from 'react';
import type { Config } from '_config';


interface Props {
  readonly title: string;
  readonly description: string;
  children?: never;
}

const production: boolean = process.env.NODE_ENV === 'production';

const fonts = '@font-face{font-family:Canela;font-style:normal;font-weight:700;font-display:swap;src:url("/fonts/Canela-Bold.woff") format("woff")}@font-face{font-family:International;font-style:normal;font-weight:300;font-display:swap;src:url("/fonts/NB-International-Pro-Light.ttf") format("truetype")}@font-face{font-family:International;font-style:normal;font-weight:700;font-display:swap;src:url("/fonts/NB-International-Pro-Bold.ttf") format("truetype")}@font-face{font-family:\'Arial CLS\';src:local(\'Arial\');ascent-override:80%;descent-override:21%}@font-face{font-family:\'Times CLS\';src:local(\'Times New Roman\');ascent-override:97.5%;descent-override:24%}html{font:300 18px International,\'Arial CLS\',sans-serif;scroll-behavior:smooth}h1,h2,h3{font-family:Canela,\'Times CLS\',serif;font-weight:700}';

const Head: FC<Props> = ({
  title,
  description,
}) => {
  const { publicRuntimeConfig } = useConfig();
  const {
    base,
    color,
    title: cTitle,
    image,
    description: cDescription,
    author,
  } = publicRuntimeConfig as Config;

  return (
    <NextHead>
      <meta charSet="utf-8" />
      <meta
        name="viewport"
        content="width=device-width,initial-scale=1,viewport-fit=cover"
      />
      <meta
        name="theme-color"
        content={color}
      />
      <title>{title || cTitle}</title>
      <meta
        name="description"
        content={description || cDescription}
      />
      {production && (
        <>
          <link
            rel="preload"
            as="image"
            href="/img/ficc-hero.svg"
            type="image/svg+xml"
          />
          <link
            rel="preload"
            as="font"
            href="/fonts/Canela-Bold.woff"
            type="font/woff"
            crossOrigin="anonymous"
          />
          <link
            rel="preload"
            as="font"
            href="/fonts/NB-International-Pro-Light.ttf"
            type="font/ttf"
            crossOrigin="anonymous"
          />
          <link
            rel="preload"
            as="font"
            href="/fonts/NB-International-Pro-Bold.ttf"
            type="font/ttf"
            crossOrigin="anonymous"
          />
          <link
            rel="preconnect"
            href="https://www.gstatic.com"
          />
          <link
            rel="dns-prefetch"
            href="https://firebase.googleapis.com/"
          />
          <link
            rel="dns-prefetch"
            href="https://firebaseinstallations.googleapis.com/"
          />
        </>
      )}
      <style dangerouslySetInnerHTML={{ __html: fonts }} />
      <meta
        name="author"
        content={author}
      />
      <link
        rel="canonical"
        href={base}
      />
      <meta
        name="format-detection"
        content="telephone=no,email=no"
      />
      <link
        rel="apple-touch-icon"
        href="/apple-touch-icon.png"
      />
      <link
        rel="icon"
        type="image/png"
        href="/img/favicon-96x96.png"
        sizes="96x96"
      />
      <link
        rel="icon"
        type="image/svg+xml"
        href="/img/favicon.svg"
      />
      <link
        rel="mask-icon"
        href="/img/pinned-tab.svg"
        color={color}
      />
      <link
        rel="mask-icon"
        href="/img/pinned-tab.svg"
        color="#fff"
        media="(prefers-color-schema:dark)"
      />
      <link
        rel="manifest"
        href="/ficc.webmanifest"
      />
      <meta
        name="twitter:card"
        content="summary_large_image"
      />
      <meta
        name="twitter:title"
        content={title || cTitle}
      />
      <meta
        name="twitter:description"
        content={description || cDescription}
      />
      <meta
        name="twitter:image"
        content={image}
      />
      <meta
        name="twitter:image:alt"
        content={title}
      />
      <meta
        name="twitter:url"
        content={base}
      />
      <meta
        property="og:type"
        content="website"
      />
      <meta
        property="og:title"
        content={title || cTitle}
      />
      <meta
        property="og:site_name"
        content={title}
      />
      <meta
        property="og:description"
        content={description || cDescription}
      />
      <meta
        property="og:image"
        content={image}
      />
      <meta
        property="og:image:width"
        content="1200"
      />
      <meta
        property="og:image:height"
        content="628"
      />
      <meta
        property="og:image:type"
        content="image/png"
      />
      <meta
        property="og:url"
        content={base}
      />
      {production && (
        <>
          {/* Google Tag Manager */}
          <script dangerouslySetInnerHTML={
            {
              __html: `(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
        new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
        j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
        'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
        })(window,document,'script','dataLayer','GTM-WVJL22K');`,
            }
          }
          />
          {/* End Google Tag Manager */}
          <script>
          </script>
        </>
      )}
    </NextHead>
  );
};


export default Head;
