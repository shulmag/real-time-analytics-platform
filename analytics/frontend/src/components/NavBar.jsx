// Updated NavBar.jsx with refresh button
import React from 'react';
import { Link } from 'react-router-dom';
import { Navbar, Container, Nav, Button, Spinner } from 'react-bootstrap';
import { ArrowClockwise } from 'react-bootstrap-icons'; // Make sure to install react-bootstrap-icons

function NavBar({ 
  userEmail, 
  isLoggedIn, 
  onLogout, 
  lastUpdated, 
  onRefresh, 
  isRefreshing = false,
  loadingStatus = {} // Object to track loading status of each component
}) {
  const currentDate = new Date();
  const formattedDate = currentDate.toLocaleDateString('en-US', {
    month: '2-digit',
    day: '2-digit',
    year: 'numeric'
  });
  
  const formattedTime = currentDate.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: true
  });

  // Format the last updated time if providedY
  const lastUpdatedText = lastUpdated 
    ? `Updated: ${lastUpdated.toLocaleDateString('en-US', { month: '2-digit', day: '2-digit' })}, ${lastUpdated.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true })}`
    : `Updated: ${formattedDate}, ${formattedTime}`;

  return (
    <Navbar 
      bg="white" 
      expand="lg" 
      sticky="top"
      className="shadow-sm mb-0" 
      style={{ 
        borderBottom: '1px solid #eaeaea',
        padding: '0.75rem 1rem'
      }}
    >
      <Container fluid>
        <Navbar.Brand 
          as={Link} 
          to="/" 
          style={{ 
            fontWeight: '700',
            color: '#2c3e50',
            fontSize: '1.5rem',
            letterSpacing: '-0.01em'
          }}
        >
          ficc<span style={{ color: '#3182ce' }}>.ai</span>
        </Navbar.Brand>
        <Navbar.Toggle aria-controls="basic-navbar-nav" />
        <Navbar.Collapse id="basic-navbar-nav">
          <Nav className="me-auto" style={{ marginLeft: '20px', display: 'flex', alignItems: 'center' }}>
            <span>Municipal Bond Market Analytics</span>
            
            {/* Dynamic loading message */}
            {(loadingStatus.yieldCurves || loadingStatus.codValues || loadingStatus.marketMetrics) && (
              <div className="loading-text" style={{ 
                marginLeft: '10px', 
                fontSize: '0.85rem', 
                color: '#3182ce',
                fontStyle: 'italic',
                display: 'flex',
                alignItems: 'center',
                opacity: 0.8
              }}>
                <Spinner 
                  animation="border" 
                  size="sm" 
                  style={{ marginRight: '5px', width: '10px', height: '10px' }} 
                />
                <span>
                  {loadingStatus.yieldCurves && loadingStatus.marketMetrics 
                    ? 'Loading all data...' 
                    : loadingStatus.yieldCurves 
                      ? 'Loading yield curves...' 
                      : loadingStatus.marketMetrics 
                        ? 'Loading market metrics...' 
                        : loadingStatus.codValues 
                          ? 'Calculating change of day...' 
                          : 'Loading...'}
                </span>
              </div>
            )}
          </Nav>
          <div className="d-flex align-items-center">
            <div className="d-flex align-items-center" style={{ 
              marginRight: '15px', 
              padding: '6px 12px',
              border: '1px solid #e2e8f0',
              borderRadius: '6px',
              backgroundColor: '#f8fafc',
              fontSize: '0.85rem',
              color: '#64748b',
              fontWeight: '500'
            }}>
              <span>{lastUpdatedText}</span>
              <Button
                type="button"
                style={{ marginLeft: '8px' }}
                variant="outline-primary"
                size="sm"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  onRefresh();
                }}
                disabled={isRefreshing}
                title="Refresh data"
              >
                {isRefreshing ? (
                  <>
                    <Spinner animation="border" size="sm" style={{ marginRight: 4 }} />
                    <span>Refreshing...</span>
                  </>
                ) : (
                  <>
                    <ArrowClockwise size={14} style={{ marginRight: 4 }} />
                    <span>Refresh</span>
                  </>
                )}
              </Button>
            </div>
            
            {isLoggedIn ? (
              <div className="d-flex align-items-center">
                <div style={{
                  marginLeft: '10px',
                  padding: '6px 12px',
                  backgroundColor: '#f8fafc',
                  borderRadius: '6px',
                  border: '1px solid #e2e8f0',
                  fontSize: '0.9rem',
                  color: '#4a5568',
                  display: 'flex',
                  alignItems: 'center',
                  marginRight: '10px'
                }}>
                  <span style={{ marginRight: '5px' }}>Signed in as:</span>
                  <span style={{
                    color: '#3182ce',
                    fontWeight: '500'
                  }}>
                    {userEmail || 'User'}
                  </span>
                </div>
                <Button 
                  variant="outline-secondary"
                  size="sm"
                  onClick={onLogout}
                  style={{
                    borderRadius: '6px',
                    padding: '6px 12px',
                    fontSize: '0.9rem'
                  }}
                >
                  Sign Out
                </Button>
              </div>
            ) : (
              <Button 
                as={Link}
                to="/login"
                style={{
                  padding: '8px 16px',
                  backgroundColor: '#3182ce',
                  border: 'none',
                  borderRadius: '6px',
                  transition: 'all 0.2s ease'
                }}
              >
                Sign In
              </Button>
            )}
          </div>
        </Navbar.Collapse>
      </Container>
    </Navbar>
  );
}

export default NavBar;