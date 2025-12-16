/*
 * @Date: 2021-09-29 13:05:31 
 */


import React from 'react';
import { Card } from 'react-bootstrap';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Label
} from 'recharts';

function RealtimeYieldCurve({ yield_data, yesterday_data }) {
  if (!yield_data || !yield_data.x || !yield_data.yield) {
    return <p>No yield data available.</p>;
  }

  const todayMap = new Map(yield_data.x.map((m, i) => [m, yield_data.yield[i]]));
  const ydayMap = new Map(
    yesterday_data?.x?.map((m, i) => [m, yesterday_data.yield[i]]) || []
  );

  const allMaturities = Array.from(
    new Set([...todayMap.keys(), ...ydayMap.keys()])
  ).sort((a, b) => a - b);

  const chartData = allMaturities.map((maturity) => ({
    maturity,
    today: todayMap.get(maturity),
    yesterday: ydayMap.get(maturity)
  }));

  return (
    <div className="p-3">
      <Card className="shadow-sm w-100 mb-4" style={{ maxWidth: '100vw' }}>
        <Card.Header className="bg-light py-2 d-flex justify-content-between align-items-center">
          <h6 className="mb-0">Real Time Yield Curve (Investment Grade Market)</h6>
        </Card.Header>
        <Card.Body className="p-3" style={{ height: '400px', width: '100%' }}>
          <ResponsiveContainer>
            <LineChart 
              data={chartData}
              margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="maturity"
                type="number"
                domain={[0, 30]}
                ticks={[0, 5, 10, 15, 20, 25, 30]}
                tick={{ fontSize: 12 }}
                tickFormatter={(v) => `${v}`}
                interval={0}
              >
                <Label
                  value="Years to Maturity"
                  offset={-10}
                  position="insideBottom"
                  style={{ fill: '#000', fontSize: 15 }}
                />
              </XAxis>
              <YAxis
                domain={[0, 6]}
                ticks={[0, 1, 2, 3, 4, 5, 6]}
                tick={{ fontSize: 12 }}
                tickFormatter={(v) => v.toString()}
              >
                <Label
                  angle={-90}
                  position="insideLeft"
                  offset={10}
                  value="Yield (%)"
                  style={{ fill: '#000', fontSize: 15 }}
                />
              </YAxis>
              <Tooltip
                labelFormatter={(label) => `Maturity: ${Number(label).toFixed(1)} years`}
                content={({ payload, label }) => {
                  if (!payload || !payload.length) return null;
                
                  const dataPoint = payload[0]?.payload;
                  const today = dataPoint?.today;
                  const yesterday = dataPoint?.yesterday;
                
                  const delta = typeof today === 'number' && typeof yesterday === 'number'
                    ? ((today - yesterday) * 100).toFixed(1)
                    : null;
                
                  const deltaColor = delta > 0 ? 'red' : delta < 0 ? 'green' : 'gray';
                
                  return (
                    <div style={{ backgroundColor: 'white', padding: '8px', border: '1px solid #ccc' }}>
                      <div>Maturity: {Number(label).toFixed(1)} years</div>
                      <div style={{ color: '#3182ce' }}>Yield: {today?.toFixed(3)}%</div>
                      {delta !== null && (
                        <div style={{ color: deltaColor }}>
                          Δ {delta > 0 ? '+' : ''}{delta} bps
                        </div>
                      )}
                    </div>
                  );
                }}
              />
              <Line
                type="monotone"
                dataKey="today"
                stroke="#3182ce"
                strokeWidth={2}
                dot={false}
                name="Today"
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </Card.Body>
      </Card>
    </div>
  );
}

export default RealtimeYieldCurve;