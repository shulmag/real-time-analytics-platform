import React from 'react'
import 'bootstrap/dist/css/bootstrap.css'
import ReactDOM from 'react-dom/client'
import './index.css'
import App from './App'
import reportWebVitals from './reportWebVitals'
// import firebase from 'firebase/compat/app'

import { initializeApp } from 'firebase/app'
import { getAuth } from 'firebase/auth'

// import { BrowserRouter } from 'react-router-dom'

const app = initializeApp({
    apiKey: 'AIzaSyAmNhC6vHOEVjoNjOlGsUkc_pR4dSx6eGg',
    authDomain: 'eng-reactor-287421.firebaseapp.com',
    projectId: 'eng-reactor-287421',
    storageBucket: 'eng-reactor-287421.appspot.com',
    messagingSenderId: '964018767272',
    appId: '1:964018767272:web:8c3d149ba061461819899d',
    Persistence:'SESSION'
})

const auth = getAuth(app)

// window.firebase = firebase

const root = ReactDOM.createRoot(document.getElementById('root'))
root.render(
    <React.StrictMode>
        <App />
    </React.StrictMode>
)

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();

// removing manifest.json error: https://stackoverflow.com/questions/72415961/manifest-json-error-in-line-1-in-react-js
// app.use(express.static(path.join(__dirname,'client','build')))
// app.get('/*', (req, res) => { res.sendFile(path.join(__dirname ,'/FRONTEND/public/index.html')); })
