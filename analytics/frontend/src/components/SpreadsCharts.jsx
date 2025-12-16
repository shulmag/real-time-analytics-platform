import React, { useEffect, useState, useMemo } from "react";
import { Card, Alert } from "react-bootstrap";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from "recharts";

const COLOR_1MM = "#3182ce";   // blue for 1mm bonds
const COLOR_100 = "#2f855a";   // green for 100 bonds
const Y_TICK_STEP = 0.1;

// Build ticks covering a [min,max] range
function makeTicksFromRange(min, max, step = Y_TICK_STEP) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return { ticks: [], domain: ["auto", "auto"] };
  }
  const floor = Math.floor(min / step) * step;
  const ceil  = Math.ceil(max / step) * step;

  const ticks = [];
  const round2 = v => Math.round(v * 100) / 100;
  for (let v = floor; v <= ceil + 1e-9; v = round2(v + step)) {
    ticks.push(round2(v));
  }
  if (ticks.length < 3) {
    ticks.unshift(round2(floor - step));
    ticks.push(round2(ceil + step));
  }
  return { ticks, domain: [ticks[0], ticks[ticks.length - 1]] };
}

// ---------- DATE HELPERS (timezone-safe) ----------

// Normalize any date-like string (RFC1123, ISO, etc.) to YYYY-MM-DD in UTC
function toISODate(dateLike) {
  const d = new Date(dateLike);
  if (Number.isNaN(d)) return null;
  const y = d.getUTCFullYear();
  const m = String(d.getUTCMonth() + 1).padStart(2, "0");
  const day = String(d.getUTCDate()).padStart(2, "0");
  return `${y}-${m}-${day}`; // e.g., 2025-09-08
}

const MONTH_ABBR = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];

// Expect a YYYY-MM-DD string; render compact label like "Sep 8"
const formatDateLabel = (isoYmd) => {
  if (!isoYmd) return "";
  const [y, m, d] = isoYmd.split("-").map(Number);
  if (!y || !m || !d) return "";
  return `${MONTH_ABBR[m - 1]} ${d}`;
};

// Parse "$0.297" -> 0.297 (robust to commas)
function parseDollar(v) {
  if (v == null) return NaN;
  if (typeof v === "number") return v;
  if (typeof v === "string") {
    const cleaned = v.replace(/\$/g, "").replace(/,/g, "");
    const num = parseFloat(cleaned);
    return Number.isFinite(num) ? num : NaN;
  }
  return NaN;
}

/** Normalize one quantity’s rows into one-per-day, keeping the latest time within a day */
const normalizeSeries = (arr) => {
  const byDate = new Map();
  for (const item of arr || []) {
    // UPDATED: normalize the API date (e.g., "Mon, 08 Sep 2025 00:00:00 GMT") -> "2025-09-08"
    const dateKey = toISODate(item?.date);
    if (!dateKey) continue;

    const dollarValue = parseDollar(item?.avgSpreadDollar);

    // The Map handles duplicates by keeping the last one seen for a given date.
    byDate.set(dateKey, {
      dateKey,                           // normalized YYYY-MM-DD
      date: formatDateLabel(dateKey),    // label "Sep 8"
      value: Number.isFinite(dollarValue) ? dollarValue : 0,
      cusips: Number(item?.numCusips) || 0,
    });
  }
  return [...byDate.values()].sort((a,b) => a.dateKey.localeCompare(b.dateKey));
};

/** Merge 1mm and 100 arrays on dateKey → single array for <LineChart data={...}> */
const mergeTwo = (s1mm, s100) => {
  const map = new Map();
  for (const x of s1mm) {
    map.set(x.dateKey, {
      dateKey: x.dateKey,
      date: x.date,
      s1mm: x.value,
      cusips1mm: x.cusips
    });
  }
  for (const y of s100) {
    const cur = map.get(y.dateKey) || { dateKey: y.dateKey, date: y.date };
    cur.s100 = y.value;
    cur.cusips100 = y.cusips;
    map.set(y.dateKey, cur);
  }
  return [...map.values()].sort((a,b) => a.dateKey.localeCompare(b.dateKey));
};

function SpreadsChart({
  // props for 1mm bonds
  spreadsData5Y_1mm = [],
  spreadsData10Y_1mm = [],
  loading5Y_1mm = false,
  loading10Y_1mm = false,
  error5Y_1mm = null,
  error10Y_1mm = null,

  // props for 100 bonds
  spreadsData5Y_100 = [],
  spreadsData10Y_100 = [],
  loading5Y_100 = false,    
  loading10Y_100 = false, 
  error5Y_100 = null, 
  error10Y_100 = null, 
}) {
  // merged arrays keyed by date
  const [merged5Y, setMerged5Y] = useState([]);
  const [merged10Y, setMerged10Y] = useState([]);

  // Build merged datasets (normalized dates)
  useEffect(() => {
    const a = normalizeSeries(spreadsData5Y_1mm);
    const b = normalizeSeries(spreadsData5Y_100);
    setMerged5Y(mergeTwo(a, b));
  }, [spreadsData5Y_1mm, spreadsData5Y_100]);

  useEffect(() => {
    const a = normalizeSeries(spreadsData10Y_1mm);
    const b = normalizeSeries(spreadsData10Y_100);
    setMerged10Y(mergeTwo(a, b));
  }, [spreadsData10Y_1mm, spreadsData10Y_100]);

  // Shared Y-axis across all
  const { sharedTicks, sharedDomain } = useMemo(() => {
    const values = [
      ...merged5Y.map(d => d.s1mm).filter(Number.isFinite),
      ...merged5Y.map(d => d.s100).filter(Number.isFinite),
      ...merged10Y.map(d => d.s1mm).filter(Number.isFinite),
      ...merged10Y.map(d => d.s100).filter(Number.isFinite),
    ];
    if (!values.length) return { sharedTicks: [], sharedDomain: ["auto", "auto"] };
    const min = Math.min(...values);
    const max = Math.max(...values);
    const { ticks, domain } = makeTicksFromRange(min, max, Y_TICK_STEP);
    return { sharedTicks: ticks, sharedDomain: domain };
  }, [merged5Y, merged10Y]);

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload || !payload.length) return null;
    return (
      <div className="bg-white p-3 border rounded shadow-sm">
        <p className="fw-bold mb-2">{label}</p>
        {payload.map((p) => (
          <p key={p.dataKey} className="mb-1" style={{ color: p.color }}>
            {p.name}: ${Number(p.value).toFixed(3)}
          </p>
        ))}
      </div>
    );
  };

  const ChartCard = ({ title, dataMerged, loading, error }) => (
    <Card className="shadow-sm mb-4">
      <Card.Header className="bg-light">
        <h6 className="mb-0" style={{ fontSize: "15px" }}>{title}</h6>
      </Card.Header>
      <Card.Body style={{ minHeight: '650px' }}>
        {loading ? (
          <div className="d-flex flex-column align-items-center justify-content-center h-100 p-5">
          </div>
        ) : error ? (
          <Alert variant="danger" className="m-0">
            Error loading spread data: {error}
          </Alert>
        ) : !dataMerged.length ? (
          <Alert variant="info" className="m-0">
            No spread data available for the selected period.
          </Alert>
        ) : (
          <>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart
                data={dataMerged} // ONE merged dataset
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" angle={-45} textAnchor="end" height={60} />
                <YAxis
                  ticks={sharedTicks}
                  domain={sharedDomain}
                  tickFormatter={(v) => `$${Number(v).toFixed(3)}`}
                  width={80}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend verticalAlign="top" align="right" wrapperStyle={{ paddingTop: "10px" }} />
                {/* 100 bonds */}
                <Line
                  type="monotone"
                  dataKey="s100"
                  stroke={COLOR_100}
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  name="Dollar Spread (100 bonds)"
                  connectNulls
                />
                {/* 1mm bonds */}
                <Line
                  type="monotone"
                  dataKey="s1mm"
                  stroke={COLOR_1MM}
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  name="Dollar Spread (1mm bonds)"
                  connectNulls
                />
              </LineChart>
            </ResponsiveContainer>

            {/* Summary */}
            <div className="mt-4 p-3 bg-light rounded">
              <h6 className="mb-3">Summary Statistics</h6>

              <div className="table-responsive">
                <table className="table table-sm mb-0">
                  <thead>
                    <tr>
                      <th></th>
                      <th style={{ color: COLOR_100 }}>100 bonds</th>
                      <th style={{ color: COLOR_1MM }}>1mm bonds</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="text-muted" style={{ paddingLeft: "1rem" }}>Latest Dollar Spread</td>
                      <td style={{ color: COLOR_100 }}>
                        {dataMerged.at(-1)?.s100 != null
                          ? `$${Number(dataMerged.at(-1).s100).toFixed(3)}`
                          : "-"}
                      </td>
                      <td style={{ color: COLOR_1MM }}>
                        {dataMerged.at(-1)?.s1mm != null
                          ? `$${Number(dataMerged.at(-1).s1mm).toFixed(3)}`
                          : "-"}
                      </td>
                    </tr>
                    <tr>
                      <td className="text-muted" style={{ paddingLeft: "1rem" }}>Number of CUSIPs</td>
                      <td style={{ color: COLOR_100 }}>
                        {dataMerged.at(-1)?.cusips100 != null
                          ? Number(dataMerged.at(-1).cusips100).toLocaleString()
                          : "-"}
                      </td>
                      <td style={{ color: COLOR_1MM }}>
                        {dataMerged.at(-1)?.cusips1mm != null
                          ? Number(dataMerged.at(-1).cusips1mm).toLocaleString()
                          : "-"}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}
      </Card.Body>
    </Card>
  );

  return (
    <div className="p-3">
      <ChartCard
        title="Average Bid/Ask Spreads for Investment Grade Bonds - 5 Year Maturity (1mm and 100 bonds)"
        dataMerged={merged5Y}
        loading={loading5Y_1mm || loading5Y_100}
        error={error5Y_1mm || error5Y_100}
      />

      <ChartCard
        title="Average Bid/Ask Spreads for Investment Grade Bonds - 10 Year Maturity (1mm and 100 bonds)"
        dataMerged={merged10Y}
        loading={loading10Y_1mm || loading10Y_100}
        error={error10Y_1mm || error10Y_100}
      />
    </div>
  );
}

export default SpreadsChart;
