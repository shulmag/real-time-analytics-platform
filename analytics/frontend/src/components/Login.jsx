import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { Container, Form, Button, Card, Alert, Spinner, Nav, Tab } from 'react-bootstrap';
import { auth } from '../services/auth';
import { EMAIL_ONLY_AUTH_ENABLED } from '../config';

// Define components outside the main component to prevent recreation on each render
const EmailFieldComponent = React.memo(({ email, handleEmailChange, autoFocusEmail }) => (
  <Form.Group className="mb-3">
    <Form.Label style={{ 
      fontSize: '0.9rem',
      fontWeight: '500',
      color: '#4a5568'
    }}>
      Email Address
    </Form.Label>
    <Form.Control
      type="email"
      value={email}
      onChange={handleEmailChange}
      required
      placeholder="your@email.com"
      autoFocus={autoFocusEmail}
      autoComplete="email"
      style={{
        borderRadius: '6px',
        padding: '10px 12px',
        border: '1px solid #e2e8f0',
        fontSize: '0.95rem'
      }}
      className="shadow-sm"
    />
  </Form.Group>
));

function Login({ onLoginSuccess }) {
  // States for both authentication methods
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [authMode, setAuthMode] = useState('email-link'); // 'password' or 'email-link' - default to magic link
  const [linkSent, setLinkSent] = useState(false);
  const [fieldFocus, setFieldFocus] = useState('email'); // Track which field should have focus
  
  const passwordInputRef = useRef(null);
  const emailInputRef = useRef(null);

  // Complete the email link sign-in process  
  const completeEmailLinkSignIn = useCallback(async (emailToUse, link) => {
    console.log("Completing email link sign-in");
    console.log("Email:", emailToUse);
    console.log("Link:", link);
    
    setError(null);
    setLoading(true);
    
    try {
      console.log("Calling auth.signInWithLink...");
      const result = await auth.signInWithLink(emailToUse, link);
      console.log("Sign-in successful!", result);
      
      // Clear the URL parameters after successful sign-in
      // This is important to avoid confusion with the routing logic
      if (window.history.pushState) {
        const newUrl = window.location.protocol + "//" + window.location.host + window.location.pathname;
        window.history.pushState({path: newUrl}, '', newUrl);
      }
      // Adding a slight delay to ensure state updates before redirecting
      setTimeout(() => {
        if (onLoginSuccess) {
          console.log("Calling onLoginSuccess with email:", result.email);
          onLoginSuccess(result.email);
          
          // Force redirect to the root path to trigger dashboard
          window.location.href = "/";
        }
      }, 100);
    } catch (err) {
      console.error("Error completing sign-in:", err);
      
      // Handle specific errors
      if (err.message && err.message.includes("email is already registered")) {
        // If the email is already registered, show a message and focus the password field
        setError("This email is already registered. Please sign in with your password.");
        setAuthMode('password');
        setFieldFocus('password');
      } else if (err.message && err.message.includes("link is invalid or has expired")) {
        setError("The sign-in link is invalid or has expired. Please request a new link.");
      } else {
        setError(`Error completing sign-in: ${err.message || err.toString()}`);
      }
    } finally {
      setLoading(false);
    }
  }, [onLoginSuccess]);
  
  // Handle the email link sign-in
  const handleEmailLinkSignIn = useCallback((savedEmail, link) => {
    if (savedEmail) {
      setEmail(savedEmail);
      console.log("Completing email link sign-in with saved email");
      
      // Move this to a setTimeout to avoid React hook order issues
      setTimeout(() => {
        completeEmailLinkSignIn(savedEmail, link);
      }, 0);
    } else {
      // If email not found in storage, we'll need to ask the user
      console.log("No saved email found. Asking user for email.");
      setAuthMode('email-link');
      setError('Please enter the email you used to request the login link');
      setFieldFocus('email');
    }
  }, [setEmail, setAuthMode, setError, completeEmailLinkSignIn]);
  
  // Check if we have an email link login attempt
  useEffect(() => {
    console.log("Checking for email link sign-in...");
    const isEmailLinkSignIn = auth.isSignInLink(window.location.href);
    console.log("Is email link sign-in:", isEmailLinkSignIn);
    console.log("Current URL:", window.location.href);
    
    if (isEmailLinkSignIn) {
      console.log("Email link sign-in detected!");
      
      // Extract email from URL if available (some email clients add it)
      const urlParams = new URLSearchParams(window.location.search);
      const emailFromUrl = urlParams.get('email');
      
      // Get the email from localStorage or URL
      const savedEmail = auth.getEmailFromStorage() || emailFromUrl;
      console.log("Saved email from storage or URL:", savedEmail);
      
      // Handle the sign-in
      handleEmailLinkSignIn(savedEmail, window.location.href);
    } else {
      console.log("No email link sign-in detected");
    }
  }, [handleEmailLinkSignIn]);

  // Handle traditional email/password login
  const handlePasswordSignIn = useCallback(async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const result = await auth.signIn(email, password);
      if (onLoginSuccess) {
        onLoginSuccess(result.email);
      }
    } catch (err) {
      console.error("Password sign-in error:", err);
      
      // Provide user-friendly error messages
      if (err.code === 'auth/wrong-password' || err.code === 'auth/user-not-found') {
        setError('Invalid email or password. Please try again.');
      } else if (err.code === 'auth/too-many-requests') {
        setError('Too many failed login attempts. Please try again later or reset your password.');
      } else if (err.code === 'auth/operation-not-allowed') {
        setError('Email/password sign-in is not enabled. Please contact support.');
      } else {
        setError(err.message || 'Failed to sign in. Please try again later.');
      }
    } finally {
      setLoading(false);
    }
  }, [email, password, onLoginSuccess]);

  // Handle passwordless email sign in
  const handleEmailLinkRequest = useCallback(async (e) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);
    setLoading(true);
    try {
      const result = await auth.sendSignInLink(email);
      setLinkSent(true);
      
      // Handle both standard and fallback methods
      if (result.message && result.message.includes('fallback')) {
        setSuccess(`We've sent a secure sign-in link to ${email}. Please check your inbox and click the link to access your account.`);
      } else {
        setSuccess(`We've sent a secure sign-in link to ${email}. Please check your inbox to complete the sign-in process.`);
      }
    
    } catch (err) {
      console.error("Email link auth error:", err);
      
      // Provide a more user-friendly error message
      // Handle errors related to Dynamic Links more gracefully
      if (err.message && (
          err.message.includes("Dynamic Links domain not configured") || 
          err.message.includes("Dynamic Links not activated"))) {
        // Just try again - the system will fall back to a regular email link
        setError("There was an issue sending the link. Please try again.");
        setLinkSent(false);
      } 
      // Production configuration errors
      else if (err.code === 'auth/operation-not-allowed') {
        setError('Email link sign-in is not enabled. Please use password sign-in or contact support.');
      } else if (err.code === 'auth/dynamic-link-not-activated') {
        setError('Email link sign-in is not fully configured. Please use password sign-in or contact support.');
      } else if (err.code === 'auth/invalid-continue-uri') {
        setError('The application URL configuration is invalid. Please contact support.');
      } else if (err.code === 'auth/missing-continue-uri') {
        setError('The application URL configuration is missing. Please contact support.');
      } else if (err.code === 'auth/unauthorized-continue-uri') {
        setError('The application URL is not authorized. Please contact support.');
      } else {
        setError(err.message || 'Failed to send login link. Please try again later.');
      }
    } finally {
      setLoading(false);
    }
  }, [email]);

  // Handle tab changes
  const handleTabChange = useCallback((key) => {
    setAuthMode(key);
    // Set focus based on the selected tab
    setFieldFocus(key === 'password' ? 'email' : 'email');
  }, []);

  // Common email input field for both auth methods
  // Using useCallback to prevent recreating the onChange handler on each render
  const handleEmailChange = useCallback((e) => {
    setEmail(e.target.value);
  }, []);
  
  const handlePasswordChange = useCallback((e) => {
    setPassword(e.target.value);
    // Keep focus on password field after state update
    setFieldFocus('password');
  }, []);

  // Password field focus handler
  const handlePasswordFocus = useCallback(() => {
    setFieldFocus('password');
  }, []);

  // Email field focus handler
  const handleEmailFocus = useCallback(() => {
    setFieldFocus('email');
  }, []);

  // Handle focus management with useEffect
  useEffect(() => {
    if (fieldFocus === 'password' && passwordInputRef.current) {
      passwordInputRef.current.focus();
    } else if (fieldFocus === 'email' && emailInputRef.current) {
      emailInputRef.current.focus();
    }
  }, [fieldFocus, authMode]);

  // Focus password field when switching to password tab
  useEffect(() => {
    if (authMode === 'password') {
      // Small delay to ensure the tab pane is visible first
      setTimeout(() => {
        setFieldFocus('password');
      }, 50);
    }
  }, [authMode]);

  // Create memoized components to prevent recreation on each render
  const EmailField = useMemo(() => (
    <EmailFieldComponent 
      email={email} 
      handleEmailChange={handleEmailChange} 
      autoFocusEmail={fieldFocus === 'email'} 
      ref={emailInputRef}
      onFocus={handleEmailFocus}
    />
  ), [email, handleEmailChange, fieldFocus, handleEmailFocus]);

  // Email Link UI - shows either the form to request link or confirmation message
  const EmailLinkUI = useMemo(() => {
    if (linkSent) {
      return (
        <div className="text-center my-4">
          <div className="mb-4 p-3 bg-light rounded">
            <h5 className="mb-3">Check your inbox</h5>
            <p className="mb-2">We've sent a secure sign-in link to:</p>
            <p className="font-weight-bold mb-3">{email}</p>
            <p className="small text-muted mb-0">Please check your inbox and click the link to sign in to your ficc.ai account. The link will expire in 15 minutes.</p>
          </div>
          <Button 
            variant="outline-secondary" 
            size="sm"
            onClick={() => {
              setLinkSent(false);
              setSuccess(null);
              setFieldFocus('email');
            }}
          >
            Use a different email
          </Button>
        </div>
      );
    }
    // Form to request email link
    return (
      <Form onSubmit={handleEmailLinkRequest}>
        <Form.Group className="mb-3">
          <Form.Label style={{ 
            fontSize: '0.9rem',
            fontWeight: '500',
            color: '#4a5568'
          }}>
            Email Address
          </Form.Label>
          <Form.Control
            type="email"
            value={email}
            onChange={handleEmailChange}
            required
            placeholder="your@email.com"
            autoFocus={fieldFocus === 'email' && authMode === 'email-link'}
            autoComplete="email"
            ref={emailInputRef}
            onFocus={handleEmailFocus}
            style={{
              borderRadius: '6px',
              padding: '10px 12px',
              border: '1px solid #e2e8f0',
              fontSize: '0.95rem'
            }}
            className="shadow-sm"
          />
        </Form.Group>
        <div className="d-grid gap-2">
          <Button
            type="submit"
            disabled={loading}
            style={{
              width: '100%',
              padding: '10px',
              backgroundColor: '#3182ce',
              border: 'none',
              borderRadius: '6px',
              fontWeight: '500',
              transition: 'all 0.2s ease'
            }}
            onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#2c5282'}
            onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#3182ce'}
          >
            {loading ? (
              <>
                <Spinner
                  as="span"
                  animation="border"
                  size="sm"
                  role="status"
                  aria-hidden="true"
                  className="me-2"
                />
                Sending secure link...
              </>
            ) : (
              'Sign in with Email'
            )}
          </Button>
        </div>
      </Form>
    );
  }, [linkSent, email, loading, handleEmailChange, handleEmailFocus, fieldFocus, authMode, handleEmailLinkRequest]);

  // Password login UI with memoized handlers
  const PasswordLoginUI = useMemo(() => (
    <Form onSubmit={handlePasswordSignIn}>
      <Form.Group className="mb-3">
        <Form.Label style={{ 
          fontSize: '0.9rem',
          fontWeight: '500',
          color: '#4a5568'
        }}>
          Email Address
        </Form.Label>
        <Form.Control
          type="email"
          value={email}
          onChange={handleEmailChange}
          required
          placeholder="your@email.com"
          autoFocus={fieldFocus === 'email' && authMode === 'password'}
          autoComplete="email"
          ref={emailInputRef}
          onFocus={handleEmailFocus}
          style={{
            borderRadius: '6px',
            padding: '10px 12px',
            border: '1px solid #e2e8f0',
            fontSize: '0.95rem'
          }}
          className="shadow-sm"
        />
      </Form.Group>
      <Form.Group className="mb-4">
        <Form.Label style={{ 
          fontSize: '0.9rem',
          fontWeight: '500',
          color: '#4a5568'
        }}>
          Password
        </Form.Label>
        <Form.Control
          type="password"
          ref={passwordInputRef}
          value={password}
          onChange={handlePasswordChange}
          onFocus={handlePasswordFocus}
          required
          placeholder="••••••••"
          autoFocus={fieldFocus === 'password'}
          autoComplete="current-password"
          style={{
            borderRadius: '6px',
            padding: '10px 12px',
            border: '1px solid #e2e8f0',
            fontSize: '0.95rem'
          }}
          className="shadow-sm"
        />
      </Form.Group>
      <Button
        type="submit"
        disabled={loading}
        style={{
          width: '100%',
          padding: '10px',
          backgroundColor: '#3182ce',
          border: 'none',
          borderRadius: '6px',
          fontWeight: '500',
          transition: 'all 0.2s ease'
        }}
        onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#2c5282'}
        onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#3182ce'}
      >
        {loading ? (
          <>
            <Spinner
              as="span"
              animation="border"
              size="sm"
              role="status"
              aria-hidden="true"
              className="me-2"
            />
            Signing in...
          </>
        ) : (
          'Sign In'
        )}
      </Button>
    </Form>
  ), [email, password, loading, handleEmailChange, handlePasswordChange, handleEmailFocus, handlePasswordFocus, fieldFocus, authMode, handlePasswordSignIn]);

  return (
    <Container className="d-flex justify-content-center align-items-center" style={{ minHeight: '70vh' }}>
      <Card style={{ 
        width: '400px', 
        borderRadius: '10px',
        boxShadow: '0 4px 12px rgba(0,0,0,0.08)'
      }}>
        <Card.Header 
          className="text-center py-3" 
          style={{ 
            backgroundColor: '#f7fafc',
            borderBottom: '1px solid #e2e8f0'
          }}
        >
          <h4 style={{ 
            margin: 0, 
            color: '#2d3748',
            fontWeight: '600'
          }}>
            ficc<span style={{ color: '#3182ce' }}>.ai</span> Login
          </h4>
        </Card.Header>
        <Card.Body className="px-4 py-4">
          {error && (
            <Alert variant="danger">{error}</Alert>
          )}
          {success && (
            <Alert variant="success">{success}</Alert>
          )}
          {EMAIL_ONLY_AUTH_ENABLED ? (
            /* If email-only auth is enabled, show both options */
            <Tab.Container activeKey={authMode} onSelect={handleTabChange}>
              <Nav variant="tabs" className="mb-3">
                <Nav.Item style={{ flex: 1 }}>
                  <Nav.Link 
                    eventKey="password" 
                    className="text-center"
                    style={{ 
                      fontSize: '0.9rem', 
                      fontWeight: authMode === 'password' ? '600' : '400' 
                    }}
                  >
                    Password Sign In
                  </Nav.Link>
                </Nav.Item>
                <Nav.Item style={{ flex: 1 }}>
                  <Nav.Link 
                    eventKey="email-link" 
                    className="text-center"
                    style={{ 
                      fontSize: '0.9rem', 
                      fontWeight: authMode === 'email-link' ? '600' : '400' 
                    }}
                  >
                    Email Sign In
                  </Nav.Link>
                </Nav.Item>
              </Nav>
              <Tab.Content>
                <Tab.Pane eventKey="password">
                  {PasswordLoginUI}
                </Tab.Pane>
                <Tab.Pane eventKey="email-link">
                  {EmailLinkUI}
                </Tab.Pane>
              </Tab.Content>
            </Tab.Container>
          ) : (
            /* If email-only auth is disabled, only show password login */
            PasswordLoginUI
          )}
          <div className="mt-4 text-center">
            <small className="text-muted">
              Beta - contact ficc.ai to learn more
            </small>
          </div>
        </Card.Body>
      </Card>
    </Container>
  );
}

export default Login;