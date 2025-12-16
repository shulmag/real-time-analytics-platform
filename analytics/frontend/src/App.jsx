import React, { useLayoutEffect, useState, useEffect, useCallback, useRef } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import NavBar from './components/NavBar';
import Dashboard from './components/Dashboard';
import Login from './components/Login';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';
import { auth } from './services/auth';
import { onAuthStateChanged } from 'firebase/auth';
import { auth as firebaseAuth } from './services/firebase';
import { EMAIL_ONLY_AUTH_ENABLED } from './config';

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [userEmail, setUserEmail] = useState('');
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(new Date());
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState({
    yieldCurves: true,
    codValues: true,
    marketMetrics: true
  });
  
  // Reference to the Dashboard component to trigger data refresh
  const dashboardRef = useRef();

  useEffect(() => {
    // Set up Firebase auth state listener
    const unsubscribe = onAuthStateChanged(firebaseAuth, (user) => {
      if (user) {
        // User is signed in
        console.log('Firebase auth state changed: User is signed in', user.email);
        setIsLoggedIn(true);
        setUserEmail(user.email);
        
        // Still store in localStorage for backwards compatibility
        const userData = { email: user.email };
        localStorage.setItem('user', JSON.stringify(userData));
        
        // Clean up any email sign-in flow markers
        sessionStorage.removeItem('is_email_link_flow');
      } else {
        // User is signed out
        console.log('Firebase auth state changed: User is signed out');
        
        // Check if we're currently in an email link sign-in flow
        const isInEmailLinkFlow = sessionStorage.getItem('is_email_link_flow');
        
        // Only check localStorage as fallback if we're not in an email link flow
        if (!isInEmailLinkFlow) {
          const userData = localStorage.getItem('user');
          if (userData) {
            const { email } = JSON.parse(userData);
            setIsLoggedIn(true);
            setUserEmail(email);
          } else {
            setIsLoggedIn(false);
            setUserEmail('');
          }
        } else {
          // If we're in an email link flow, don't try to use localStorage cache
          setIsLoggedIn(false);
          setUserEmail('');
        }
      }
      setLoading(false);
    });
    
    // Cleanup subscription on unmount
    return () => unsubscribe();
  }, []);
  
  // Check for email link sign-in
  useEffect(() => {
    // Check if this is an email link sign-in
    const checkEmailLink = EMAIL_ONLY_AUTH_ENABLED && auth.isSignInLink(window.location.href);
    
    if (checkEmailLink) {
      console.log("Email link sign-in detected in App.jsx");
      // Store that we're in a link sign-in process
      sessionStorage.setItem('is_email_link_flow', 'true');
    }
  }, []);

  const handleLogin = (email) => {
    setIsLoggedIn(true);
    setUserEmail(email);
    
    // Store user data for session persistence
    const userData = { email };
    localStorage.setItem('user', JSON.stringify(userData));
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setUserEmail('');
    localStorage.removeItem('user');
    
    // Clear the token cache if available
    if (window.apiServiceInstance && typeof window.apiServiceInstance.clearTokenCache === 'function') {
      window.apiServiceInstance.clearTokenCache();
    }
    
    auth.signOut();
  };

  const handleRefresh = useCallback(() => {
    if (isRefreshing) return;

    const y = window.scrollY; // save current scroll
    setIsRefreshing(true);

    if (dashboardRef.current?.refreshData) {
      dashboardRef.current.refreshData()
        .then(() => {
          setLastUpdated(new Date());
          // Restore scroll after React has painted AND the event loop is clear
          requestAnimationFrame(() => {
            setTimeout(() => window.scrollTo(0, y), 0);
          });
        })
        .catch((error) => {
          console.error('Error refreshing data:', error);
        })
        .finally(() => {
          setIsRefreshing(false);
        });
    } else {
      // Fallback if ref not ready
      setTimeout(() => {
        setLastUpdated(new Date());
        requestAnimationFrame(() => window.scrollTo(0, y));
        setIsRefreshing(false);
      }, 500);
    }
  }, [isRefreshing]);

  if (loading) {
    return (
      <div className="d-flex justify-content-center align-items-center vh-100">
        <div className="spinner-border text-primary" role="status">
          <span className="visually-hidden">Loading...</span>
        </div>
      </div>
    );
  }

  // Check if this is an email link sign-in
  const isEmailLinkSignIn = EMAIL_ONLY_AUTH_ENABLED && auth.isSignInLink(window.location.href);

  return (
    <Router>
      <div className="app d-flex flex-column min-vh-100">
        <NavBar 
          isLoggedIn={isLoggedIn} 
          userEmail={userEmail}
          onLogout={handleLogout}
          lastUpdated={lastUpdated}
          onRefresh={handleRefresh}
          isRefreshing={isRefreshing}
          loadingStatus={loadingStatus}
        />
        
        <main className="flex-grow-1">
          <Routes>
            {/* If we have an email link, always show the Login component to process it */}
            {isEmailLinkSignIn ? (
              <Route path="*" element={<Login onLoginSuccess={handleLogin} />} />
            ) : (
              <>
                {/* Regular routes when not processing email links */}
                <Route path="/login" element={
                  isLoggedIn ? <Navigate to="/" /> : <Login onLoginSuccess={handleLogin} />
                } />
                <Route path="/" element={
                  isLoggedIn 
                    ? <Dashboard 
                        ref={dashboardRef} 
                        onDataLoaded={() => setLastUpdated(new Date())}
                        onLoadingChange={setLoadingStatus}
                      /> 
                    : <Navigate to="/login" />
                } />
                {/* Catch-all route */}
                <Route path="*" element={<Navigate to="/" />} />
              </>
            )}
          </Routes>
        </main>
        
        <footer className="bg-dark text-white text-center p-3">
          <small>© 2025 ficc.ai - Municipal Bond Analytics</small>
        </footer>
      </div>
    </Router>
  );
}

export default App;