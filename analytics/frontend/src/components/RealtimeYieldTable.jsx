import React from 'react';
import { Table, Card } from 'react-bootstrap';

function RealtimeYieldTable({ todayData, yesterdayData }) {
  if (
    !todayData || !todayData.x || !todayData.yield ||
    !yesterdayData || !yesterdayData.x || !yesterdayData.yield
  ) {
    return <p>Yield curve table data is incomplete.</p>;
  }

  const today = new Date();
  const month = String(today.getMonth() + 1).padStart(2, '0');
  const day = String(today.getDate()).padStart(2, '0');
  const year = today.getFullYear();
  const headerDate = `${month}-${day}-${year}`;

  const todayMap = new Map(todayData.x.map((m, i) => [m, todayData.yield[i]]));
  const ydayMap = new Map(yesterdayData.x.map((m, i) => [m, yesterdayData.yield[i]]));

  const allMaturities = Array.from(new Set([...todayMap.keys(), ...ydayMap.keys()]))
    .sort((a, b) => a - b);

  const baseYear = new Date().getFullYear(); // avoid hard-coding 2025

  return (
    <>
      {/* If you keep inline styles, they must be inside the returned JSX */}
      <style>{`
        .fs-7 { font-size: 0.75rem; }   /* ~12px */
        .fs-8 { font-size: 0.625rem; }  /* ~10px */
      `}</style>

      <div className="d-flex justify-content-center">
        <div style={{ width: '100%', maxWidth: '600px' }}>
          <Card.Header className="bg-light py-2 text-center mb-3">
            <h6 className="mb-0" style={{ whiteSpace: 'pre-wrap' }}>
              {`${headerDate}     Real Time Yield Curve (Investment Grade Market)`}
            </h6>
          </Card.Header>

          <Table
            striped
            bordered
            size="sm"
            className="fs-7"
            style={{ width: '100%', maxWidth: '600px' }}
          >
            <thead>
              <tr>
                <th>Years to Maturity</th>
                <th>Today (%)</th>
                <th>Previous Day (%)</th>
                <th>&#916; (bps)</th>
              </tr>
            </thead>
            <tbody>
              {allMaturities.map((maturity, idx) => {
                const todayYield = todayMap.get(maturity);
                const ydayYield = ydayMap.get(maturity);

                let delta = null;
                let deltaClass = 'text-muted';

                if (typeof todayYield === 'number' && typeof ydayYield === 'number') {
                  const diff = todayYield - ydayYield;
                  delta = (diff * 100).toFixed(1); // percent -> bps
                  if (diff > 0.0001) {
                    deltaClass = 'text-danger';
                    delta = `+${delta}`;
                  } else if (diff < -0.0001) {
                    deltaClass = 'text-success';
                  }
                }

                return (
                  <tr key={idx}>
                    <td>{maturity} &nbsp;&nbsp;({baseYear + maturity})</td>
                    <td>{todayYield?.toFixed(3) ?? '-'}</td>
                    <td>{ydayYield?.toFixed(3) ?? '-'}</td>
                    <td className={deltaClass}>{delta ?? '-'}</td>
                  </tr>
                );
              })}
            </tbody>
          </Table>
        </div>
      </div>
    </>
  );
}

export default RealtimeYieldTable;
