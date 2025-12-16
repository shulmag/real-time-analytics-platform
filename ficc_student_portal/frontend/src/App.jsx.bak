// App.jsx with EULA modal
import { useState } from 'react'
import axios from 'axios'
import './App.css'

// API endpoint - change this to match your deployed server
// const API_URL = 'http://localhost:8000/api/apply'
const API_URL = 'https://students-964018767272.us-central1.run.app/api/apply'

function App() {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [submitted, setSubmitted] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [showEulaModal, setShowEulaModal] = useState(false)
  const [eulaAccepted, setEulaAccepted] = useState(false)
  const [formData, setFormData] = useState(null)

  const validateAndShowEula = (e) => {
    e.preventDefault()
    
    // Basic email validation on the client side
    const academicDomains = ['.edu', '.ac.', 'university', 'college', 'school']
    const isAcademic = academicDomains.some(domain => email.toLowerCase().includes(domain))
    
    if (!isAcademic) {
      setError('Please use your academic email address.')
      return
    }
    
    if (!name.trim()) {
      setError('Please enter your name.')
      return
    }
    
    // Store form data for later submission
    setFormData({ name, email })
    
    // Show EULA modal
    setShowEulaModal(true)
    setError('')
  }

  const handleEulaAccept = () => {
    setEulaAccepted(true)
  }

  const handleEulaCancel = () => {
    setShowEulaModal(false)
    setEulaAccepted(false)
  }

  const handleSubmitAfterEula = async () => {
    if (!eulaAccepted || !formData) return
    
    setLoading(true)
    setShowEulaModal(false)
    
    try {
      // Send data to the server
      const response = await axios.post(API_URL, {
        name: formData.name,
        email: formData.email,
        eulaAccepted: true
      })
      
      console.log('Response:', response.data)
      setSubmitted(true)
    } catch (err) {
      console.error('Error submitting form:', err)
      
      // Handle error responses from the server
      if (err.response && err.response.data) {
        setError(err.response.data.detail || 'An error occurred. Please try again later.')
      } else {
        setError('Network error. Please check your connection and try again.')
      }
    } finally {
      setLoading(false)
      setEulaAccepted(false)
    }
  }

  return (
    <div className="app-container">
      <div className="content-wrapper">
        <header>
          <h1>ficc.ai Campus Access</h1>
        </header>
        
        <main>
          <div className="description">
            <p>
              ficc.ai is proud to support the next generation of innovators through our <em>Campus Access</em> program. 
              Full-time undergraduate and graduate students—whether currently enrolled in school or completing an 
              internship—are eligible for <strong>free access to the ficc.ai User Interface</strong> in renewable 
              3-month periods. It's our way of empowering students to explore, learn, and create with powerful 
              tools at no cost.
            </p>
          </div>
          
          {!submitted ? (
            <div className="form-container">
              <h2>Apply for Free Access</h2>
              <form onSubmit={validateAndShowEula}>
                <div className="form-group">
                  <label htmlFor="name">Full Name</label>
                  <input 
                    type="text" 
                    id="name"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    required
                    placeholder="Enter your full name"
                    disabled={loading}
                  />
                </div>
                
                <div className="form-group">
                  <label htmlFor="email">Academic Email</label>
                  <input 
                    type="email" 
                    id="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                    placeholder="Your academic email address"
                    disabled={loading}
                  />
                  <small>Please use your academic email (.edu, university email, etc.)</small>
                  {error && <div className="error-message">{error}</div>}
                </div>
                
                <button 
                  type="submit" 
                  className={`submit-btn ${loading ? 'loading' : ''}`}
                  disabled={loading}
                >
                  {loading ? 'Submitting...' : 'Apply Now'}
                </button>
              </form>
            </div>
          ) : (
            <div className="success-message">
              <h2>Thank You!</h2>
              <p>Your application has been submitted. We'll review your details and contact you shortly with your access credentials.</p>
            </div>
          )}
        </main>
        
        <footer>
          <p>&copy; {new Date().getFullYear()} ficc.ai. All rights reserved.</p>
        </footer>
      </div>
      
      {/* EULA Modal */}
      {showEulaModal && (
        <div className="modal-overlay">
          <div className="modal-container">
            <div className="modal-header">
              <h2>ficc.ai Campus Access Terms and Conditions</h2>
              <p className="modal-date">Effective Date: March 27, 2025</p>
            </div>
            
            <div className="modal-content">
              <p>By registering for student access to ficc.ai, you agree to the following terms and conditions:</p>
              
              <div className="term-section">
                <h3>1. Eligibility and Term</h3>
                <ul>
                  <li>Access is available to full-time undergraduate and graduate students only.</li>
                  <li>Access is granted in 3-month renewable periods and may be extended as long as you remain a full-time student.</li>
                  <li>You must use a valid academic email address and may be required to provide proof of enrollment.</li>
                </ul>
              </div>
              
              <div className="term-section">
                <h3>2. Permitted Use</h3>
                <ul>
                  <li>Access is for your individual academic or personal educational use only.</li>
                  <li>You may not use ficc.ai for any commercial, professional, or trading-related purposes.</li>
                  <li>Usage must be reasonable and aligned with personal or coursework-related learning. Automated scraping or excessive usage is not permitted.</li>
                </ul>
              </div>
              
              <div className="term-section">
                <h3>3. Prohibited Use</h3>
                <ul>
                  <li>You may not share your login credentials or access with any other person.</li>
                  <li>Use of ficc.ai to support professional research, trading strategies, consulting, or paid academic projects is strictly prohibited.</li>
                  <li>ficc.ai may monitor usage patterns and revoke access if use appears inconsistent with these terms.</li>
                </ul>
              </div>
              
              <div className="term-section">
                <h3>4. Right to Revoke</h3>
                <ul>
                  <li>ficc.ai may suspend or revoke access at any time and for any reason, without notice.</li>
                  <li>You must delete any restricted or proprietary data upon termination of access.</li>
                </ul>
              </div>
              
              <div className="term-section">
                <h3>5. Use of Data</h3>
                <ul>
                  <li>Certain data incorporates proprietary content from ficc.ai and third-party providers, including but not limited to S&P Global. Access to and use of this data is strictly limited to internal business purposes and must comply with all restrictions.</li>
                </ul>
              </div>
              
              <div className="term-section">
                <h3>6. Restrictions on Use</h3>
                <ul>
                  <li>The Data may not be copied, distributed, sublicensed, or otherwise transferred to any third party.</li>
                  <li>The Data may not be modified, reverse-engineered, or used to create derivative works that replicate original data sources. No use of the Data is permitted for benchmarking, creating indices, or in connection with any AI or machine learning applications without prior written consent.</li>
                </ul>
              </div>
              
              <div className="term-section">
                <h3>7. No Warranties and Liability</h3>
                <ul>
                  <li>The Data is provided "as is" without any warranties, express or implied.</li>
                  <li>ficc.ai, S&P Global, and any other third-party providers disclaim any liability for inaccuracies, errors, or omissions in the Data.</li>
                  <li>Users assume full responsibility for any reliance on the Data and agree that ficc.ai and its licensors shall not be liable for any direct or indirect damages arising from its use.</li>
                </ul>
              </div>
              
              <div className="term-section">
                <h3>8. Termination of Access</h3>
                <ul>
                  <li>ficc.ai reserves the right to revoke access to the Data at any time if terms of use are violated.</li>
                  <li>Users must certify deletion of restricted data upon termination of access.</li>
                </ul>
              </div>
              
              <div className="term-section">
                <h3>9. Compliance</h3>
                <ul>
                  <li>Users must comply with all applicable laws, regulations, and contractual obligations regarding data access and use.</li>
                </ul>
              </div>
            </div>
            
            <div className="modal-actions">
              <div className="eula-checkbox">
                <input 
                  type="checkbox" 
                  id="accept-eula" 
                  checked={eulaAccepted} 
                  onChange={() => setEulaAccepted(!eulaAccepted)} 
                />
                <label htmlFor="accept-eula">I have read and agree to the Terms and Conditions</label>
              </div>
              
              <div className="modal-buttons">
                <button className="cancel-btn" onClick={handleEulaCancel}>Cancel</button>
                <button 
                  className="accept-btn" 
                  disabled={!eulaAccepted} 
                  onClick={handleSubmitAfterEula}
                >
                  Accept & Submit
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App