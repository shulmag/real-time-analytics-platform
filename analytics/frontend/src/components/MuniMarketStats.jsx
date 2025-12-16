/*
*/

import React, { useEffect, useState } from 'react';
import { Row, Col, Card } from 'react-bootstrap';
import { apiService } from '../services/apiService';
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend
} from 'recharts';

const nf  = new Intl.NumberFormat('en-US');
const cf0 = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });
const dlabel = (iso) => {
  const [y, m, d] = iso.split('-');
  return `${m}/${d}`;
};

export default function MuniMarketStats({ onLoadingChange }) {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState({});
  const [error, setError] = useState(null);

  const [chartLoading, setChartLoading] = useState(true);
  const [chartData, setChartData] = useState([]);
  const [chartError, setChartError] = useState(null);

  // NEW: Top issues state
  const [topIssues, setTopIssues] = useState(null);
  const [topIssuesLoading, setTopIssuesLoading] = useState(true);
  const [topIssuesError, setTopIssuesError] = useState(null);
  const [issueType, setIssueType] = useState('all');      // 'all' | 'seasoned' | 'new'
  const [bucket, setBucket] = useState('current');        // 'current' | 'previous'

  // Initial loads (market stats + 10d chart)
  useEffect(() => {
    (async () => {
      onLoadingChange?.(true); // Report that loading has started
      try {
        const res = await apiService.getMuniMarketStats();
        setData(res || {});
      } catch (e) {
        console.error('Muni market stats error:', e);
        setError('Failed to load muni market stats');
      } finally {
        setLoading(false);
      }
    })();

    (async () => {
      try {
        const res = await apiService.getMuniMarketStats10d();
        setChartData(Array.isArray(res) ? res : []);
      } catch (e) {
        console.error('Muni market stats 10d error:', e);
        setChartError('Failed to load 10-day chart');
      } finally {
        setChartLoading(false);
        onLoadingChange?.(false); // Report loading finished after the last fetch completes
      }
    })();
  }, [onLoadingChange]);

  // NEW: Top issues fetch (refetch when issueType changes)
  useEffect(() => {
    (async () => {
      try {
        setTopIssuesLoading(true);
        const res = await apiService.getMuniTopIssues({
          issue_type: issueType === 'all' ? undefined : issueType,
        });
        setTopIssues(res || null);
      } catch (e) {
        console.error('Muni top issues error:', e);
        setTopIssuesError('Failed to load top issues');
      } finally {
        setTopIssuesLoading(false);
      }
    })();
  }, [issueType]);

  const Cards = () => (
    <>
      {/* CSS Grid for the three cards */}
      <section className="mms-grid mt-3">
        {/* 1×1 — top-left */}
        <div className="tile trading-overview">
          <Card className="h-100 shadow-sm">
            <Card.Body className="p-3 d-flex flex-column">
              <h6 className="card-title mb-2">Trading Overview</h6>
              <div className="mb-1 small text-muted">
                <span>Total Number of Trades Today: </span>
                <strong>{nf.format(data.total_trades_today ?? 0)}</strong>
              </div>
              <div className="mb-1 small text-muted">
                <span>Total Number of Trades This Year: </span>
                <strong>{nf.format(data.total_trades_this_year ?? 0)}</strong>
              </div>
              <div className="mb-1 small text-muted">
                <span>Total Volume of Trades Today: </span>
                <strong>{cf0.format(data.total_volume_today ?? 0)}</strong>
              </div>
            </Card.Body>
          </Card>
        </div>

        {/* 1×1 — bottom-left */}
        <div className="tile customer-flow">
          <Card className="h-100 shadow-sm">
            <Card.Body className="p-3 d-flex flex-column">
              <h6 className="card-title mb-2">Customer Flow</h6>
              <div className="mb-1 small text-muted">
                <span>Average Par Traded Today: </span>
                <strong>{cf0.format(data.avg_par_today ?? 0)}</strong>
              </div>
              <div className="mb-1 small text-muted">
                <span>Number of Customer Bought Trades Today: </span>
                <strong>{nf.format(data.customer_bought_trades_today ?? 0)}</strong>
              </div>
              <div className="mb-1 small text-muted">
                <span>Number of Customer Sold Trades Today: </span>
                <strong>{nf.format(data.customer_sold_trades_today ?? 0)}</strong>
              </div>
            </Card.Body>
          </Card>
        </div>

        {/* 2×2 — right side (replaced with Top 10 Issues) */}
        <div className="tile most-active">
          <Card className="h-100 shadow-sm">
            <Card.Body className="p-3 d-flex flex-column fs-6">
              <div className="d-flex justify-content-between align-items-center mb-2">
                <h6 className="card-title mb-0">Most Traded Issues</h6>
                <div className="d-flex gap-2">
                  {/* Current / Previous toggle */}
                  <div className="btn-group btn-group-sm" role="group" aria-label="Bucket">
                    <button
                      type="button"
                      className={`btn ${bucket === 'current' ? 'btn-dark' : 'btn-outline-secondary'}`}
                      onClick={() => setBucket('current')}
                    >
                      Today
                    </button>
                    <button
                      type="button"
                      className={`btn ${bucket === 'previous' ? 'btn-dark' : 'btn-outline-secondary'}`}
                      onClick={() => setBucket('previous')}
                    >
                      Yesterday
                    </button>
                  </div>
                  {/* Issue type filter */}
                  <select
                    className="form-select form-select-sm"
                    style={{ width: 140 }}
                    value={issueType}
                    onChange={(e) => setIssueType(e.target.value)}
                    aria-label="Issue type filter"
                  >
                    <option value="all">All types</option>
                    <option value="seasoned">Seasoned</option>
                    <option value="new">New</option>
                  </select>
                </div>
              </div>

              {/* As-of date */}
              <div className="text-muted small mb-3">
                {topIssues &&
                  (bucket === 'current'
                    ? (topIssues.current_as_of_date ? `As of ${topIssues.current_as_of_date}` : '')
                    : (topIssues.previous_as_of_date ? `As of ${topIssues.previous_as_of_date}` : ''))}
              </div>

              {/* Body */}
              {topIssuesError ? (
                <div className="alert alert-danger mb-0">{topIssuesError}</div>
              ) : topIssuesLoading ? (
                null
              ) : !topIssues || !Array.isArray(bucket === 'current' ? topIssues.current : topIssues.previous) ? (
                <div className="alert alert-info mb-0">No data available.</div>
              ) : (
                <div className="table-responsive">
                  <table className="table table-sm align-middle fs-7 table-tight">
                    <thead className="table-light">
                      <tr>
                        <th style={{ width: 48 }}>Rank</th>
                        <th style={{ width: 140 }}>CUSIP</th>
                        <th>Description</th>
                        <th style={{ width: 80 }}>Type</th>
                        <th style={{ width: 100 }} className="text-end">Trades</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(bucket === 'current' ? topIssues.current : topIssues.previous)
                        .slice(0, 10)
                        .map((r) => (
                          <tr key={`${r.rank}-${r.cusip}`}>
                            <td>{r.rank}</td>
                            <td className="font-monospace">{r.cusip}</td>
                            <td>{r.security_description}</td>
                            <td className="text-capitalize">{r.issue_type}</td>
                            <td className="text-end">{nf.format(r.trade_count)}</td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Footer source echo (optional) */}
            </Card.Body>
          </Card>
        </div>
      </section>

      {/* Inline CSS for the grid (works in Vite/React) */}
      <style>{`
        .mms-grid {
          display: grid;
          grid-template-columns: repeat(6, minmax(0, 1fr));
          grid-template-rows: repeat(2, minmax(160px, auto));
          gap: 12px;
          width: 100%;
        }
        /* Placement per your spec */
        .trading-overview { grid-column: 1 / span 2; grid-row: 1 / span 1; }
        .customer-flow    { grid-column: 1 / span 2; grid-row: 2 / span 1; }
        .most-active      { grid-column: 3 / span 4; grid-row: 1 / span 2; }

        /* Make cards fill tiles nicely */
        .tile .card { height: 100%; }
        .tile .card-body { height: 100%; }

        /* Background like your old Row bg-light */
        .mms-grid { background: var(--bs-light, #f8f9fa); padding: 12px; border-radius: 6px; }

        .fs-7 {font-size: 0.75rem; /* ~12px */}
        .fs-8 {font-size: 0.625rem; /* ~10px */}
        .table-tight td {
            line-height: 1.4;  /* adjust up/down until it looks right */
            padding-top: 0.4rem;  /* optional: tweak row padding */
            padding-bottom: 0.4rem;
            }

'''
# may be added later 

        .stats-grid {
          display: grid;
          /* Define two columns: first one sizes to its content, second takes the rest */
          grid-template-columns: auto 1fr;
          /* Add some space between rows and columns */
          gap: 0.25rem 1rem; /* 0.25rem for rows, 1rem for columns */
          align-items: center;
        }
'''          
        /* Responsive: stack on small screens */
        @media (max-width: 992px) {
          .mms-grid {
            grid-template-columns: 1fr;
            grid-template-rows: none;
          }
          .trading-overview,
          .customer-flow,
          .most-active {
            grid-column: auto;
            grid-row: auto;
          }
        }
      `}</style>
    </>
  );

  const Chart = () => (
    <Row className="mx-0 mt-3">
      <Col xs={12}>
        <Card className="shadow-sm">
          <Card.Header className="bg-light py-2 d-flex justify-content-between align-items-center">
            <h6 className="mb-0">Daily Trade Count</h6>
          </Card.Header>
          <Card.Body className="p-3">
            {chartError ? (
              <div className="alert alert-danger mb-0">{chartError}</div>
            ) : chartLoading ? (
              null
            ) : chartData.length === 0 ? (
              <div className="alert alert-info mb-0">No data available.</div>
            ) : (
              <div style={{ width: '100%', height: 320 }}>
                <ResponsiveContainer>
                  <BarChart data={chartData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }} barSize={60}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" tickFormatter={dlabel} interval={0} height={30} />
                    <YAxis />
                    <Tooltip
                      formatter={(val, name) => [nf.format(val), name]}
                      labelFormatter={(v) => `Date: ${v}`}
                      content={({ active, payload, label }) => {
                        if (!active || !payload) return null;
                        const sorted = [...payload].reverse();
                        return (
                          <div className="custom-tooltip bg-white p-2 border rounded">
                            <div>{`Date: ${label}`}</div>
                            {sorted.map((entry, index) => (
                              <div key={`item-${index}`} style={{ color: entry.color }}>
                                {`${entry.name}: ${nf.format(entry.value)}`}
                              </div>
                            ))}
                          </div>
                        );
                      }}
                    />
                    <Legend layout="vertical" verticalAlign="middle" align="right" 
                      wrapperStyle={{
                        paddingLeft: "30px" // Adjust this value to get the desired space
                      }}
                    />
                    {/* Palette: D = orange, S = green, P = blue */}
                    <Bar dataKey="P" stackId="trades" name="Bid Side (P)"    fill="#4E79A7" />
                    <Bar dataKey="S" stackId="trades" name="Offer Side (S)"   fill="#59A14F" />
                    <Bar dataKey="D" stackId="trades" name="Inter-Dealer (D)" fill="#F28E2B" />
                    
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </Card.Body>
        </Card>
      </Col>
    </Row>
  );

  if (loading) {
    return null
  }
  if (error) return <div className="my-3 alert alert-danger">{error}</div>;

  return (
    <>
      <Cards />
      <Chart />
    </>
  );
}