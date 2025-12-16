import { initializeApp } from 'firebase/app';
import { getAuth, signInWithEmailAndPassword, signOut } from 'firebase/auth';

const firebaseConfig = {
    apiKey: 'AIzaSyAmNhC6vHOEVjoNjOlGsUkc_pR4dSx6eGg',
    authDomain: 'eng-reactor-287421.firebaseapp.com',
    projectId: 'eng-reactor-287421',
    storageBucket: 'eng-reactor-287421.appspot.com',
    messagingSenderId: '964018767272',
    appId: '1:964018767272:web:8c3d149ba061461819899d',
    Persistence: 'SESSION'
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

export const login = (email, password) => {
  return signInWithEmailAndPassword(auth, email, password);
};

export const logout = () => {
  return signOut(auth);
};

export { auth };
