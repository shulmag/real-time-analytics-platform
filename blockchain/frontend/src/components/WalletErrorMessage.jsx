// WalletErrorMessage.jsx
import React from 'react';

/**
 * WalletErrorMessage component for displaying wallet-specific messages
 * including connection prompts and error messages
 */
const WalletErrorMessage = ({ error, isConnected }) => {
  // If connected, don't show any message
  if (isConnected) return null;
  
  // If there's an error, show the error message
  if (error) {
    // Simple string error
    if (typeof error === 'string') {
      return (
        <div className="alert alert-danger" role="alert">
          <i className="bi bi-exclamation-triangle me-2"></i>
          {error}
        </div>
      );
    }

    // Object-based error with type and message
    const { type, message } = error;
    
    // Default action button - install Freighter
    let actionButton = (
      <a
        href="https://www.freighter.app/"
        target="_blank"
        rel="noopener noreferrer"
        className="btn btn-outline-danger mt-2"
      >
        <i className="bi bi-download me-2"></i>
        Install Freighter
      </a>
    );
    
    // Additional guidance based on error type
    let guidance = "Freighter is required to interact with the Stellar blockchain.";
    
    // Customize based on error type
    switch (type) {
      case 'wallet_missing':
        guidance = "You need to install the Freighter wallet extension to connect to Stellar.";
        break;
        
      case 'wallet_locked':
        guidance = "Your Freighter wallet is installed but currently locked. Please open the extension and unlock it with your password to continue.";
        actionButton = (
          <button
            className="btn btn-outline-primary mt-2"
            onClick={() => window.open('chrome-extension://jmmhkeiopnimkchaoagfgkeoalcidgkb/index.html')}
          >
            <i className="bi bi-unlock me-2"></i>
            Open Freighter
          </button>
        );
        break;
        
      case 'wallet_not_connected':
        guidance = "Your Freighter wallet is installed and unlocked but not connected to this application.";
        actionButton = (
          <button
            className="btn btn-outline-primary mt-2"
            onClick={() => window.location.reload()}
          >
            <i className="bi bi-link-45deg me-2"></i>
            Connect Wallet
          </button>
        );
        break;
        
      case 'user_rejected':
      case 'permission_denied':
        guidance = "This application needs permission to access your Stellar account. No transactions will be made without your explicit approval.";
        actionButton = (
          <button
            className="btn btn-outline-primary mt-2"
            onClick={() => window.location.reload()}
          >
            <i className="bi bi-arrow-clockwise me-2"></i>
            Try Again
          </button>
        );
        break;
        
      case 'extension_error':
        guidance = "There's an issue with your Freighter wallet extension. Try reloading the page or reinstalling the extension.";
        actionButton = (
          <div>
            <button
              className="btn btn-outline-primary mt-2 me-2"
              onClick={() => window.location.reload()}
            >
              <i className="bi bi-arrow-clockwise me-2"></i>
              Reload Page
            </button>
            <a
              href="https://www.freighter.app/"
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-outline-secondary mt-2"
            >
              <i className="bi bi-download me-2"></i>
              Reinstall Freighter
            </a>
          </div>
        );
        break;
        
      case 'timeout':
        guidance = "Your wallet didn't respond in time. This could mean the wallet is locked or not properly installed.";
        actionButton = (
          <div>
            <button
              className="btn btn-outline-primary mt-2 me-2"
              onClick={() => window.location.reload()}
            >
              <i className="bi bi-arrow-clockwise me-2"></i>
              Try Again
            </button>
            <a
              href="https://www.freighter.app/"
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-outline-secondary mt-2"
            >
              <i className="bi bi-download me-2"></i>
              Reinstall Freighter
            </a>
          </div>
        );
        break;
        
      default:
        // For general errors, keep default message
        break;
    }
    
    return (
      <div className="alert alert-danger" role="alert">
        <div className="d-flex align-items-center mb-2">
          <i className="bi bi-exclamation-triangle fs-4 me-2"></i>
          <strong>{message}</strong>
        </div>
        <p className="mb-2">{guidance}</p>
        <div className="d-flex justify-content-end">
          {actionButton}
        </div>
      </div>
    );
  }
  
  // Default case - general wallet connection message (when no specific error)
  return (
    <div className="alert alert-info" role="alert">
      <div className="d-flex align-items-center">
        <i className="bi bi-wallet2 fs-4 me-3"></i>
        <div>
          <strong>Please connect your Freighter wallet to update CUSIP prices</strong>
          <p className="mb-0 mt-1">
            Freighter wallet is required to interact with the Stellar blockchain and record prices.
          </p>
        </div>
      </div>
      <div className="d-flex justify-content-end mt-2">
        <a
          href="https://www.freighter.app/"
          target="_blank"
          rel="noopener noreferrer"
          className="btn btn-outline-primary me-2"
        >
          <i className="bi bi-download me-2"></i>
          Get Freighter
        </a>
      </div>
    </div>
  );
};

export default WalletErrorMessage;