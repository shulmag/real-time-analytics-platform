/*
 * @Date: 2025-03-06
 */

// src/components/AboutSection.jsx
import React, { useState, useRef, forwardRef, useImperativeHandle } from 'react';

// Using forwardRef to expose methods to parent components
const AboutSection = forwardRef((props, ref) => {
  const [expanded, setExpanded] = useState(false);
  const sectionRef = useRef(null);

  // Expose the expand method to parent components
  useImperativeHandle(ref, () => ({
    expand: () => {
      setExpanded(true);
      if (sectionRef.current) {
        sectionRef.current.scrollIntoView({ behavior: 'smooth' });
      }
    }
  }));

  return (
    <div id="about-section" className="card shadow-sm mb-4" ref={sectionRef}>
      <div className="card-header bg-light">
        <h5 className="card-title mb-0">
          About the Municipal Bond Oracle
        </h5>
      </div>
      <div className="card-body">
        <p>
          The <strong>Muni Price Oracle</strong> is a decentralized price feed for municipal bonds powered by <a href="https://ficc.ai" target="_blank" rel="noopener noreferrer">ficc.ai</a> and the Stellar blockchain. This oracle provides transparent, immutable price records for municipal bonds using Soroban smart contracts.
        </p>
        
        {!expanded ? (
          <button 
            className="btn btn-sm btn-outline-primary" 
            onClick={() => setExpanded(true)}
          >
            Learn More
          </button>
        ) : (
          <>
            <h6 className="mt-3">Key Concepts</h6>
            <ul className="mb-4">
                <li><strong>CUSIP</strong> - A unique 9-character identifier for municipal bonds (e.g., "13063D7Q5"). Each municipal bond has its own CUSIP.</li>
              <li><strong>Quantity</strong> - The face value of bonds in thousands (e.g., "100" means $100,000 in bonds). Must be in multiples of 5.</li>
              <li><strong>Trade Types</strong>:
                <ul className="mt-1">
                  <li><em>Inter-Dealer (D)</em> - Trades between bond dealers</li>
                  <li><em>Bid Side (P)</em> - Customer sell price</li>
                  <li><em>Offered Side (S)</em> - Customer buy price</li>
                </ul>
              </li>
            </ul>
            
            <h6 className="mt-3">How It Works</h6>
            <ol>
              <li><strong>Connect Wallet</strong> - Use the Freighter wallet to connect to Stellar</li>
              <li><strong>Enter CUSIP</strong> - The default CUSIP (13063D7Q5) is for a California municipal bond, or find your own on <a href="https://emma.msrb.org/" target="_blank" rel="noopener noreferrer">EMMA</a></li>
              <li><strong>Set Quantity and Trade Type</strong> - Specify how many bonds (in $1000s) and whether it's a buy, sell, or inter-dealer trade</li>
              <li><strong>Get Live Price</strong> - Click to fetch real-time price data from ficc.ai's pricing engine</li>
              <li><strong>Confirm Transaction</strong> - Review the network fee and confirm to record the price on the blockchain</li>
              <li><strong>View Price History</strong> - Use the search function in "Latest Blockchain Prices" to see historical data for any CUSIP</li>
            </ol>

            <h6 className="mt-3">Benefits of Blockchain Price Oracle</h6>
            <ul className="mb-4">
              <li><strong>Transparency</strong> - All price data is publicly visible and verifiable</li>
              <li><strong>Immutability</strong> - Once recorded, prices cannot be altered or deleted</li>
              <li><strong>Trustlessness</strong> - No need to trust a central authority for price reporting</li>
              <li><strong>Accessibility</strong> - Anyone can access historical price data for any municipal bond</li>
              <li><strong>Auditability</strong> - Complete transaction history with timestamps</li>
            </ul>

            <h6 className="mt-3">FAQ</h6>
            <div className="accordion" id="faqAccordion">
              <div className="accordion-item">
                <h2 className="accordion-header">
                  <button className="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#faqOne">
                    What is a Municipal Bond Oracle?
                  </button>
                </h2>
                <div id="faqOne" className="accordion-collapse collapse" data-bs-parent="#faqAccordion">
                  <div className="accordion-body">
                    A Municipal Bond Oracle is a trusted data feed that provides real-time municipal bond pricing information to blockchain applications. Our oracle sources pricing from ficc.ai's institutional-grade pricing service and records it immutably on the Stellar blockchain.
                  </div>
                </div>
              </div>
              
              <div className="accordion-item">
                <h2 className="accordion-header">
                  <button className="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#faqTwo">
                    Why use blockchain for bond prices?
                  </button>
                </h2>
                <div id="faqTwo" className="accordion-collapse collapse" data-bs-parent="#faqAccordion">
                  <div className="accordion-body">
                    Blockchain technology provides transparency, immutability, and auditability for price data. This creates a trusted record that can be used for reporting, compliance, and DeFi applications related to municipal bonds.
                  </div>
                </div>
              </div>
              
              <div className="accordion-item">
                <h2 className="accordion-header">
                  <button className="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#faqThree">
                    Do I need a Freighter wallet?
                  </button>
                </h2>
                <div id="faqThree" className="accordion-collapse collapse" data-bs-parent="#faqAccordion">
                  <div className="accordion-body">
                    You only need a Freighter wallet if you want to add new price data to the blockchain. If you're just viewing the data, no wallet is required. <a href="https://www.freighter.app/" target="_blank" rel="noopener noreferrer">Install Freighter</a> to participate in price updates.
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-3">
              <p>
                This Oracle is powered by <a href="https://ficc.ai" target="_blank" rel="noopener noreferrer" className="fw-bold">ficc.ai</a>, the leading provider of municipal bond pricing and analytics. 
                Visit <a href="https://ficc.ai" target="_blank" rel="noopener noreferrer">ficc.ai</a> to learn more about our comprehensive municipal bond solutions.
              </p>
              <button 
                className="btn btn-sm btn-outline-secondary" 
                onClick={() => setExpanded(false)}
              >
                Show Less
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
});

export default AboutSection;