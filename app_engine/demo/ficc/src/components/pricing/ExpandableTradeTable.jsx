/*
 * @Date: 2025-02-25
 */

import React, { useState } from 'react';
import Table from 'react-bootstrap/Table';
import Card from 'react-bootstrap/Card';
import Badge from 'react-bootstrap/Badge';
import { FaChevronDown, FaChevronRight } from 'react-icons/fa';

// This component handles both the trade history and similar bonds tables
// Example usage for trade history: 
// <ExpandableTradeTable 
//    title="Recent trade history for 64971XQM3" 
//    data={tradeHistory} 
//    type="tradeHistory" 
//    cusip={cusipForDisplay} 
// />

function ExpandableTradeTable({ title, data, type, cusip }) {
  const [expandedRows, setExpandedRows] = useState([]);

  const toggleRow = (rowId) => {
    setExpandedRows(prevExpandedRows => {
      const newExpandedRows = [...prevExpandedRows];
      const index = newExpandedRows.indexOf(rowId);
      
      if (index === -1) {
        newExpandedRows.push(rowId);
      } else {
        newExpandedRows.splice(index, 1);
      }
      
      return newExpandedRows;
    });
  };

  // Determine column structure based on table type
  const getTableHeaders = () => {
    if (type === 'tradeHistory') {
      return (
        <tr>
          <th style={styles.tableHeader}>
            <span>Trade Date</span>
          </th>
          <th style={styles.tableHeader}>Vol (k)</th>
          <th style={styles.tableHeader}>Trade Count</th>
          <th style={styles.tableHeader}>High / Low Price (%)</th>
          <th style={styles.tableHeader}>High / Low Yield (%)</th>
          <th style={styles.tableHeader}>Bid Side Vol (k)</th>
          <th style={styles.tableHeader}>Offered Side Vol (k)</th>
          <th style={styles.tableHeader}>Inter-dealer Vol (k)</th>
        </tr>
      );
    } else if (type === 'similarBonds') {
      return (
        <tr>
          <th style={styles.tableHeader}>CUSIP</th>
          <th style={styles.tableHeader}>State</th>
          <th style={styles.tableHeader}>S&P</th>
          <th style={styles.tableHeader}>Description</th>
          <th style={styles.tableHeader}>Avg Yield (%)</th>
          <th style={styles.tableHeader}>Avg Price (%)</th>
          <th style={styles.tableHeader}>Coupon</th>
          <th style={styles.tableHeader}>Maturity Date</th>
        </tr>
      );
    }
  };

  const styles = {
    card: {
      marginBottom: '2rem',
      borderRadius: '10px',
      border: '1px solid #e2e8f0',
      boxShadow: '0 2px 5px rgba(0,0,0,0.05)'
    },
    cardHeader: {
      backgroundColor: '#f7fafc',
      padding: '16px 20px',
      borderBottom: '1px solid #e2e8f0'
    },
    headerTitle: {
      color: '#2d3748',
      fontSize: '1.1rem',
      fontWeight: '600',
      margin: 0
    },
    tableContainer: {
      padding: '0.5rem'
    },
    table: {
      borderCollapse: 'separate',
      borderSpacing: 0,
      width: '100%',
      marginBottom: 0
    },
    tableHeader: {
      backgroundColor: '#edf2f7',
      color: '#4a5568',
      fontWeight: '600',
      fontSize: '0.85rem',
      padding: '12px 16px',
      borderBottom: '2px solid #e2e8f0',
      textAlign: 'left',
      verticalAlign: 'middle'
    },
    tableRow: {
      cursor: 'pointer',
      transition: 'background-color 0.2s'
    },
    tableCell: {
      padding: '12px 16px',
      borderTop: '1px solid #e2e8f0',
      fontSize: '0.9rem',
      color: '#2d3748',
      verticalAlign: 'middle'
    },
    badge: {
      padding: '4px 8px',
      fontSize: '0.75rem',
      fontWeight: '600',
      borderRadius: '4px'
    },
    expandIcon: {
      marginRight: '8px',
      fontSize: '0.85rem',
      verticalAlign: 'middle'
    },
    nestedTableContainer: {
      backgroundColor: '#f8fafc',
      padding: '12px 16px',
      borderTop: '1px solid #e2e8f0'
    },
    nestedTable: {
      margin: 0,
      fontSize: '0.85rem'
    },
    nestedTableHeader: {
      backgroundColor: '#edf2f7',
      color: '#4a5568',
      fontWeight: '600',
      fontSize: '0.8rem',
      padding: '10px 14px'
    },
    nestedTableCell: {
      padding: '10px 14px',
      fontSize: '0.85rem',
      color: '#4a5568'
    },
    emptyState: {
      padding: '2rem',
      textAlign: 'center',
      color: '#718096'
    }
  };

  // Render table rows from provided data
  const renderTableRows = () => {
    // Guard clause: if data is not an array or is empty, return nothing
    if (!data || !Array.isArray(data) || data.length === 0) {
      return null;
    }
    
    // Map through the data and render rows
    return data.map((row, index) => (
      <React.Fragment key={index}>
        <tr 
          onClick={() => toggleRow(row.id)} 
          style={{
            ...styles.tableRow,
            backgroundColor: expandedRows.includes(row.id) ? '#edf2f7' : 'white'
          }}
          onMouseOver={(e) => e.currentTarget.style.backgroundColor = expandedRows.includes(row.id) ? '#e2e8f0' : '#f7fafc'}
          onMouseOut={(e) => e.currentTarget.style.backgroundColor = expandedRows.includes(row.id) ? '#edf2f7' : 'white'}
        >
          {/* First cell with expand icon */}
          <td style={styles.tableCell}>
            {expandedRows.includes(row.id) ? 
              <FaChevronDown style={styles.expandIcon} /> : 
              <FaChevronRight style={styles.expandIcon} />
            }
            {type === 'tradeHistory' ? row.date : (row.cusip || 'Unknown')}
          </td>
          
          {/* Render appropriate cells based on table type */}
          {type === 'tradeHistory' ? (
            <>
              <td style={styles.tableCell}>{row.total ? (row.total / 1000).toLocaleString() : '0'}</td>
              <td style={styles.tableCell}>{row.count || 0}</td>
              <td style={styles.tableCell}>
                {row.high_price && row.low_price ? 
                  `${typeof row.high_price === 'number' ? row.high_price.toFixed(3) : row.high_price} / ${typeof row.low_price === 'number' ? row.low_price.toFixed(3) : row.low_price}` : 
                  'N/A'}
              </td>
              <td style={styles.tableCell}>
                {row.high_yield && row.low_yield ? 
                  `${typeof row.high_yield === 'number' ? row.high_yield.toFixed(3) : row.high_yield} / ${typeof row.low_yield === 'number' ? row.low_yield.toFixed(3) : row.low_yield}` : 
                  'N/A'}
              </td>
              <td style={styles.tableCell}>{row.dpVol ? (row.dpVol / 1000).toLocaleString() : '0'}</td>
              <td style={styles.tableCell}>{row.dsVol ? (row.dsVol / 1000).toLocaleString() : '0'}</td>
              <td style={styles.tableCell}>{row.ddVol ? (row.ddVol / 1000).toLocaleString() : '0'}</td>
            </>
          ) : (
            <>
              <td style={styles.tableCell}>{row.state || 'N/A'}</td>
              <td style={styles.tableCell}>
                {row.rating ? row.rating: 'N/A'}
              </td>
              <td style={styles.tableCell}>{row.security_description || 'N/A'}</td>
              <td style={styles.tableCell}>
                {typeof row.avg_yield === 'number' ? row.avg_yield.toFixed(3) : 'N/A'}
              </td>
              <td style={styles.tableCell}>
                {typeof row.avg_price === 'number' ? row.avg_price.toFixed(3) : 'N/A'}
              </td>
              <td style={styles.tableCell}>{row.coupon || 'N/A'}</td>
              <td style={styles.tableCell}>{row.maturity_date || 'N/A'}</td>
            </>
          )}
        </tr>
        
        {/* Expanded row with nested table */}
        {expandedRows.includes(row.id) && (
          <tr>
            <td colSpan={type === 'tradeHistory' ? 8 : 8} style={{ padding: 0 }}>
              <div style={styles.nestedTableContainer}>
                <Table striped bordered hover size="sm" style={styles.nestedTable}>
                  <thead>
                    <tr>
                      {type === 'tradeHistory' ? (
                        <>
                          <th style={styles.nestedTableHeader}>Trade Date & Time</th>
                          <th style={styles.nestedTableHeader}>Yield (%)</th>
                          <th style={styles.nestedTableHeader}>Price (%)</th>
                          <th style={styles.nestedTableHeader}>Yield to Worst Date</th>
                          <th style={styles.nestedTableHeader}>Trade Amount (k)</th>
                          <th style={styles.nestedTableHeader}>Trade Type</th>
                        </>
                      ) : (
                        <>
                          <th style={styles.nestedTableHeader}>Trade Date & Time</th>
                          <th style={styles.nestedTableHeader}>Yield (%)</th>
                          <th style={styles.nestedTableHeader}>Price (%)</th>
                          <th style={styles.nestedTableHeader}>Yield to Worst Date</th>
                          <th style={styles.nestedTableHeader}>Trade Amount (k)</th>
                          <th style={styles.nestedTableHeader}>Trade Type</th>
                        </>
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {/* Placeholder for nested table data */}
                    {/* This would be populated with the actual nested data */}
                    {row.details?.map((detail, detailIndex) => (
                      <tr key={detailIndex}>
                        {type === 'tradeHistory' ? (
                          <>
                            <td style={styles.nestedTableCell}>{detail.trade_datetime}</td>
                            <td style={styles.nestedTableCell}>{detail.yield_to_worst}</td>
                            <td style={styles.nestedTableCell}>{detail.dollar_price}</td>
                            <td style={styles.nestedTableCell}>{detail.calc_date}</td>
                            <td style={styles.nestedTableCell}>{detail.size / 1000}</td>
                            <td style={styles.nestedTableCell}>
                              {detail.trade_type === 'P'
                                ? 'Bid Side'
                                : detail.trade_type === 'S'
                                ? 'Offered Side'
                                : detail.trade_type === 'D'
                                ? 'Inter-dealer'
                                : detail.trade_type}
                            </td>
                          </>
                        ) : (
                          <>
                            <td style={styles.nestedTableCell}>{detail.trade_datetime}</td>
                            <td style={styles.nestedTableCell}>{detail.yield}</td>
                            <td style={styles.nestedTableCell}>{detail.dollar_price}</td>
                            <td style={styles.nestedTableCell}>{detail.calc_date}</td>
                            <td style={styles.nestedTableCell}>{detail.par_traded / 1000}</td>
                            <td style={styles.nestedTableCell}>
                              {detail.trade_type === 'P'
                                ? 'Bid Side'
                                : detail.trade_type === 'S'
                                ? 'Offered Side'
                                : detail.trade_type === 'D'
                                ? 'Inter-dealer'
                                : detail.trade_type}
                            </td>
                          </>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </Table>
              </div>
            </td>
          </tr>
        )}
      </React.Fragment>
    ));
  };

  return (
    <Card style={styles.card}>
      <Card.Header style={styles.cardHeader}>
        <h5 style={styles.headerTitle}>
          {title}
          <span style={{fontSize: '0.85rem', fontWeight: 'normal', color: '#718096', marginLeft: '10px'}}>
            (click a row to expand details)
          </span>
        </h5>
      </Card.Header>
      <div style={styles.tableContainer}>
        {data && data.length > 0 ? (
          <Table hover style={styles.table}>
            <thead>{getTableHeaders()}</thead>
            <tbody>{renderTableRows()}</tbody>
          </Table>
        ) : (
          <div style={styles.emptyState}>
            {type === 'tradeHistory' 
              ? `No trade history available for ${cusip}` 
              : 'No similar bonds found matching your criteria'}
          </div>
        )}
      </div>
    </Card>
  );
}

export default ExpandableTradeTable;