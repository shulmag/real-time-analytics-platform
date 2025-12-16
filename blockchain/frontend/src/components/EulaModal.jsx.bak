// src/components/EulaModal.jsx
import React, { useState, useEffect } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import './EnhancedEulaModal.css'; // Make sure to create this CSS file

const EulaModal = ({ onAccept }) => {
  const [showModal, setShowModal] = useState(false);
  const [isChecked, setIsChecked] = useState(false);

  useEffect(() => {
    // TESTING MODE: Force clear localStorage and always show EULA
    // Remove this line for production
    localStorage.removeItem('eulaAccepted');
    
    // Short delay for better UX - allows page to load first
    const timer = setTimeout(() => {
      setShowModal(true);
    }, 500);
    
    return () => clearTimeout(timer);

    // PRODUCTION CODE:
    // Remove localStorage.removeItem line above and uncomment this if check
    /*
    // Check if user has already accepted EULA
    const hasAcceptedEula = localStorage.getItem('eulaAccepted');
    
    if (!hasAcceptedEula) {
      // Short delay for better UX - allows page to load first
      const timer = setTimeout(() => {
        setShowModal(true);
      }, 500);
      
      return () => clearTimeout(timer);
    }
    */
  }, []);

  const handleAccept = () => {
    if (isChecked) {
      // Comment out these two lines during testing
      // to prevent saving acceptance to localStorage
      localStorage.setItem('eulaAccepted', 'true');
      localStorage.setItem('eulaAcceptedDate', new Date().toISOString());
      
      // Animate out
      const modalElement = document.getElementById('eulaModal');
      modalElement.classList.add('fade-out');
      
      setTimeout(() => {
        setShowModal(false);
        if (onAccept) onAccept();
      }, 300);
    }
  };

  if (!showModal) {
    return null;
  }

  return (
    <div 
      id="eulaModal"
      className="eula-modal-backdrop"
    >
      <div className="eula-modal-container">
        <div className="eula-modal-content">
          <div className="eula-modal-header">
            <div className="eula-title-container">
              <h5 className="eula-modal-title">ficc.ai Corp Third-Party Terms and Conditions</h5>
            </div>
          </div>
          
          <div className="eula-modal-body">
            <div className="eula-section">
              <h6>1. Use of Data</h6>
              <p>
                Certain data incorporates proprietary content from ficc.ai and third-party providers, including but not limited to S&P Global. Access to and use of this data is strictly limited to internal business purposes and must comply with all restrictions.
              </p>
            </div>
            
            <div className="eula-section">
              <h6>2. Restrictions on Use</h6>
              <ul>
                <li>The Data may not be copied, distributed, sublicensed, or otherwise transferred to any third party.</li>
                <li>The Data may not be modified, reverse-engineered, or used to create derivative works that replicate original data sources.</li>
                <li>No use of the Data is permitted for benchmarking, creating indices, or in connection with any AI or machine learning applications without prior written consent.</li>
              </ul>
            </div>
            
            <div className="eula-section">
              <h6>3. No Warranties and Liability</h6>
              <ul>
                <li>The Data is provided "as is" without any warranties, express or implied.</li>
                <li>ficc.ai, S&P Global, and any other third-party providers disclaim any liability for inaccuracies, errors, or omissions in the Data.</li>
                <li>Users assume full responsibility for any reliance on the Data and agree that ficc.ai and its licensors shall not be liable for any direct or indirect damages arising from its use.</li>
              </ul>
            </div>
            
            <div className="eula-section">
              <h6>4. Termination of Access</h6>
              <ul>
                <li>ficc.ai reserves the right to revoke access to the Data at any time if terms of use are violated.</li>
                <li>Users must certify deletion of restricted data upon termination of access.</li>
              </ul>
            </div>
            
            <div className="eula-section">
              <h6>5. Compliance</h6>
              <ul>
                <li>Users must comply with all applicable laws, regulations, and contractual obligations regarding data access and use.</li>
              </ul>
            </div>
            
            <div className="eula-checkbox-container">
              <label className="eula-checkbox">
                <input 
                  type="checkbox" 
                  checked={isChecked}
                  onChange={() => setIsChecked(!isChecked)}
                />
                <span className="eula-checkmark"></span>
                <span className="eula-label">I have read and agree to the Terms and Conditions</span>
              </label>
            </div>
          </div>
          
          <div className="eula-modal-footer">
            <button 
              type="button" 
              className={`eula-accept-button ${isChecked ? 'active' : 'disabled'}`}
              onClick={handleAccept}
              disabled={!isChecked}
            >
              Accept & Continue
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EulaModal;