import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'

import Pricing from './components/pricing'
import Compliance from './components/compliance'
import LoginForm from './components/login' 
import DailySchoonover from './components/dailySchoonover' 
import FiccYieldCurve from './components/ficcYieldCurve'
import ContactUs from './components/contact'
import ContactTab from './components/contactTab'
// import DownloadButton from './components/download/download'    // use this to allow an external user to download a single file from the front end

import './App.css'
import 'bootstrap/dist/css/bootstrap.min.css'

// import { OwnIDInit } from '@ownid/react'

function App() {
  return (
    <>
    {/* <OwnIDInit config={{ appId: 'fytab17pfvoepo', sdk: 'firebase'}}/> */}
    
    <Router>
      <Routes>
        <Route path='/' element={<LoginForm/>}/>
        <Route path='/login' element={<LoginForm/>}/>
        <Route path='/pricing' element={<Pricing/>}/>
        <Route path='/compliance' element={<Compliance/>}/>
        <Route path='/dailySchoonover' element={<DailySchoonover/>}/>
        <Route path='/ficcyieldcurve' element={<FiccYieldCurve/>}/>
        <Route path='/contact' element={<ContactUs/>}/>
        <Route path='/contactTab' element={<ContactTab/>}/>
        {/* The fileName below must be present in `ficc/public/files/` */}
        {/* <Route path='/files' element={<DownloadButton fileName='priced_2024_06_14_15_00_00.csv'/>}/> */}
      </Routes>  
    </Router>
    
    </>
    )  
}

export default App
