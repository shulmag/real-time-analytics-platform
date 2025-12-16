import NextDocument, {
  Html,
  Main,
  Head,
  NextScript,
} from 'next/document';
import schema from '@src/schema.json';


const production: boolean = process.env.NODE_ENV === 'production';

class Document extends NextDocument {
  render(): JSX.Element {
    return (
      <Html lang="en">
        <Head />
        <body onTouchStart={() => true} >
          <noscript>
            <iframe
              src="https://www.googletagmanager.com/ns.html?id=GTM-WVJL22K"
              height="0"
              width="0"
              style={{
                display: 'none',
                visibility: 'hidden',
              }}
            ></iframe>
          </noscript>

          <Main />
          <script
            type="application/ld+json"
            dangerouslySetInnerHTML={{ __html: JSON.stringify(schema) }}
          />
          {production && (
            <script
              dangerouslySetInnerHTML={{ __html: 'var firebaseConfig={apiKey:"AIzaSyAmNhC6vHOEVjoNjOlGsUkc_pR4dSx6eGg",authDomain:"eng-reactor-287421.firebaseapp.com",projectId:"eng-reactor-287421",storageBucket:"eng-reactor-287421.appspot.com",messagingSenderId:"964018767272",appId:"1:964018767272:web:8c3d149ba061461819899d",measurementId:"G-VKPXKKDZQT"};firebase.initializeApp(firebaseConfig),firebase.analytics();' }}
              defer
            />
          )}
          <NextScript />
        </body>
      </Html>
    );
  }
}


export default Document;
