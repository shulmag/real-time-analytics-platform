import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { auth } from './services/auth';
import Login from './components/Login';
import NavBar from './components/NavBar';
import PriceTable from './components/PriceTable';
import 'bootstrap/dist/css/bootstrap.min.css';
import Header from './components/Header';
import { Button } from './components/ui/button';
import { LogIn } from 'lucide-react';
import BlockchainPrices from './components/BlockchainPrices';

// Constants
const TRADE_TYPE_OPTIONS = [
  { key: 'S', text: 'Sell Side' },
  { key: 'P', text: 'Purchase Side' }
];

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [userEmail, setUserEmail] = useState('');
  const [loading, setLoading] = useState(true);
  const [cusip, setCusip] = useState('047870NE6');  // Default CUSIP
  const [quantity, setQuantity] = useState('100');   // Default quantity
  const [tradeType, setTradeType] = useState('S');   // Default trade type
  const [stellarAccount, setStellarAccount] = useState(null);

  const handleLogin = async () => {
    // Implement your login logic here
    // For example:
    // const user = await signInWithFirebase();
    // setIsLoggedIn(true);
    // setUserEmail(user.email);
  };

  const handleGetAndUpdatePrice = async () => {
    // ... existing price update logic
  };

  React.useEffect(() => {
    const unsubscribe = auth.onAuthStateChanged(user => {
      setIsLoggedIn(!!user);
      setUserEmail(user?.email || '');
      setLoading(false);
    });

    return unsubscribe;
  }, []);

  if (loading) {
    return <div>Loading...</div>;
  }

  return (
    <div className="min-h-screen bg-background">
      <Header 
        isLoggedIn={isLoggedIn}
        onLogin={handleLogin}
        userEmail={userEmail}
      />
      <Router>
        {isLoggedIn && <NavBar userEmail={userEmail} />}
        <Routes>
          <Route 
            path="/login" 
            element={isLoggedIn ? <Navigate to="/prices" /> : <Login />} 
          />
          <Route 
            path="/prices" 
            element={isLoggedIn ? <PriceTable /> : <Navigate to="/login" />} 
          />
          <Route 
            path="/" 
            element={<Navigate to={isLoggedIn ? "/prices" : "/login"} />} 
          />
        </Routes>
      </Router>
      
      <main className="container mx-auto px-4 py-6">
        <div className="card shadow-sm mb-4">
          <div className="card-header bg-light">
            <h5 className="card-title mb-0">Municipal Bond Price Information</h5>
          </div>
          
          <div className="card-body">
            {!stellarAccount && (
              <div className="alert alert-info mb-3" role="alert">
                <i className="bi bi-wallet2 me-2"></i>
                Please connect your Freighter wallet to update CUSIP prices
              </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* CUSIP Input with explanation */}
              <div className="space-y-2">
                <label htmlFor="cusipInput" className="block text-sm font-medium">
                  CUSIP
                </label>
                <input
                  id="cusipInput"
                  type="text"
                  className="w-full rounded-md border border-input px-3 py-2"
                  placeholder="Enter 9-character CUSIP"
                  value={cusip}
                  onChange={(e) => setCusip(e.target.value)}
                />
                <div className="text-sm text-muted-foreground">
                  <p>9-character identifier for municipal bonds</p>
                  <a 
                    href="https://emma.msrb.org/" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="text-primary hover:underline"
                  >
                    Find your local bonds on EMMA →
                  </a>
                </div>
              </div>

              {/* Quantity Input with explanation */}
              <div className="space-y-2">
                <label htmlFor="quantityInput" className="block text-sm font-medium">
                  Quantity
                </label>
                <input
                  id="quantityInput"
                  type="number"
                  className="w-full rounded-md border border-input px-3 py-2"
                  placeholder="Enter quantity in thousands"
                  value={quantity}
                  onChange={(e) => setQuantity(e.target.value)}
                  min="5"
                  step="5"
                />
                <div className="text-sm text-muted-foreground">
                  In thousands (e.g., 100 = $100,000). Must be multiples of 5.
                </div>
              </div>

              {/* Trade Type Selector */}
              <div className="col-span-2 space-y-2">
                <label htmlFor="tradeTypeSelect" className="block text-sm font-medium">
                  Trade Type
                </label>
                <select
                  id="tradeTypeSelect"
                  className="w-full rounded-md border border-input px-3 py-2"
                  value={tradeType}
                  onChange={(e) => setTradeType(e.target.value)}
                >
                  {TRADE_TYPE_OPTIONS.map((option) => (
                    <option key={option.key} value={option.key}>
                      {option.text}
                    </option>
                  ))}
                </select>
              </div>

              {/* Action Buttons */}
              <div className="col-span-2 space-y-4">
                <Button
                  className="w-full"
                  size="lg"
                  onClick={handleGetAndUpdatePrice}
                  disabled={loading || !cusip || !quantity || !stellarAccount}
                >
                  {loading ? (
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                  ) : (
                    <span className="flex items-center gap-2">
                      <svg className="h-4 w-4" viewBox="0 0 24 24">
                        <path fill="currentColor" d="M7 14l5-5 5 5H7z"/>
                      </svg>
                      Get Live Price
                    </span>
                  )}
                </Button>
                
                <div className="bg-muted p-4 rounded-md">
                  <p className="text-sm text-muted-foreground flex items-center gap-2">
                    <svg className="h-4 w-4" viewBox="0 0 24 24">
                      <path fill="currentColor" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
                    </svg>
                    To view existing prices, use the "Search Blockchain" option in the Latest Blockchain Prices section below.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <BlockchainPrices />
      </main>
    </div>
  );
}

export default App;
