import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../App';
import './MostActiveCusipBadge.css';

const MostActiveCusipBadge = ({ onSelectCusip }) => {
  const [mostActiveCusip, setMostActiveCusip] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchMostActiveCusip = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/most_active_cusip`);
        if (response.data?.status === 'success') {
          setMostActiveCusip(response.data.data);
        }
      } catch (err) {
        console.warn('Could not fetch most active CUSIP:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchMostActiveCusip();
  }, []);

  if (loading || !mostActiveCusip) {
    return null;
  }

  // Truncate long descriptions to keep the badge compact
  const truncateDescription = (desc) => {
    if (!desc) return '';
    return desc.length > 55 ? desc.substring(0, 55) + '...' : desc;
  };

  return (
    <div className="mt-2 most-active-badge">
      <div className="d-flex align-items-center">
        <span className="badge bg-light text-dark border border-primary me-2">
          <i className="bi bi-graph-up-arrow text-primary me-1"></i>
          Most traded yesterday
        </span>
        <button 
          type="button" 
          className="btn btn-link btn-sm p-0 text-decoration-none me-2"
          onClick={() => onSelectCusip && onSelectCusip(mostActiveCusip.cusip)}
          title={mostActiveCusip.description}  // Full description as tooltip
        >
          <strong>{mostActiveCusip.cusip}</strong>
          <i className="bi bi-arrow-right-circle-fill ms-1 text-primary"></i>
        </button>
        <span className="small text-muted d-inline-flex align-items-center">
          <span className="me-2">({mostActiveCusip.trade_count} trades)</span>
          <span className="cusip-description" title={mostActiveCusip.description}>
            {truncateDescription(mostActiveCusip.description)}
          </span>
        </span>
      </div>
    </div>
  );
};

export default MostActiveCusipBadge;