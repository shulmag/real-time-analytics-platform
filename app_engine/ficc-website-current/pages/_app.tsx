import type { FC } from 'react';
import type { AppProps } from 'next/app';
import '../src/styles/core/index.scss';
import '../src/styles/global/main.scss';


const App: FC<AppProps> = ({ Component, pageProps }) => <Component {...pageProps} />;


export default App;
