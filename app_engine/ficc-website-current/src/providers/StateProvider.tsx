import {
  createContext,
  useContext,
  FC,
} from 'react';


interface Props {
  readonly slug: string;
  readonly children: JSX.Element | JSX.Element[];
}

interface State {
  readonly slug: string;
}


const StateContext = createContext<string>('');

const StateProvider: FC<Props> = ({ slug, children }) => (
  <StateContext.Provider value={slug}>
    {children}
  </StateContext.Provider>
);

const state = (): State => ({ slug: useContext(StateContext) });


export default StateProvider;
export { state };
