// src/components/FeeConfirmationModal.jsx
import React from 'react';
import PropTypes from 'prop-types';

const FeeConfirmationModal = ({ 
    show,
    fees, 
    onConfirm, 
    onCancel, 
    isLoading,
    priceData 
}) => {
    if (!show) return null;

    // Check if fees exists before trying to access its properties
    const hasFees = fees && typeof fees === 'object';
    
    return (
        <div className="modal show" style={{display: 'block', backgroundColor: 'rgba(0,0,0,0.5)'}}>
            <div className="modal-dialog">
                <div className="modal-content">
                    <div className="modal-header">
                        <h5 className="modal-title">Confirm Transaction</h5>
                    </div>
                    <div className="modal-body">
                        <div className="alert alert-info mb-3">
                            <i className="bi bi-info-circle me-2"></i>
                            The final transaction cost will be shown in your Freighter wallet for confirmation.
                        </div>
                        
                        <div className="mb-3">
                            <h6>Estimated Transaction Fees:</h6>
                            {hasFees ? (
                                <ul className="list-unstyled">
                                    <li>Network Fee: {fees.base_fee_xlm.toFixed(7)} XLM</li>
                                    <li>Resource Fee: {2*fees.resource_fee_xlm.toFixed(7)} XLM</li>
                                    <li className="fw-bold mt-2">Total: {2*fees.total_xlm.toFixed(7)} XLM</li>
                                    <li className="text-muted small">
                                    (≈ ${(fees.total_xlm * 2 * 0.3).toFixed(2)} USD)
                                    </li>
                                </ul>
                            ) : (
                                <p>Calculating fees...</p>
                            )}
                        </div>

                    </div>
                    <div className="modal-footer">
                        <button 
                            className="btn btn-secondary" 
                            onClick={onCancel}
                            disabled={isLoading}
                        >
                            Cancel
                        </button>
                        <button 
                            className="btn btn-primary" 
                            onClick={onConfirm}
                            disabled={isLoading || !hasFees}
                        >
                            {isLoading ? (
                                <>
                                    <span className="spinner-border spinner-border-sm me-2" />
                                    Processing...
                                </>
                            ) : (
                                'Continue to Wallet'
                            )}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

FeeConfirmationModal.propTypes = {
    show: PropTypes.bool.isRequired,
    fees: PropTypes.shape({
        base_fee_xlm: PropTypes.number,
        resource_fee_xlm: PropTypes.number,
        total_xlm: PropTypes.number,
        approximate_usd: PropTypes.number
    }),
    onConfirm: PropTypes.func.isRequired,
    onCancel: PropTypes.func.isRequired,
    isLoading: PropTypes.bool,
    priceData: PropTypes.shape({
        price: PropTypes.number,
        yield: PropTypes.number
    })
};

export default FeeConfirmationModal;