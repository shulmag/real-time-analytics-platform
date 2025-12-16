import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';

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

// Initialize Firebase Authentication and get a reference to the service
const auth = getAuth(app);

// Export the auth object
export { auth };