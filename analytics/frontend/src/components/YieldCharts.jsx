/*
 * @Date: 2025-07-26
 */

/**
 * YieldCharts component for displaying IG Yield Curves.
 * Yesterday's chart has Y-axis on the left, today's chart has Y-axis on the right.
 * Margins tuned to avoid clipping of the 4 PM label, Y-axis labels moved outward, and charts positioned closer together.
 * NEW: Y-axis ticks are computed at fixed 0.5 basis point intervals (0.005%) per chart.
 * NEW: Plot window is 9:35–4:00 PM ET (minutes 5–390). Points before 9:35 are dropped.
 */

import React, { useEffect, useState } from "react";
import { Card, Col, Row, Spinner, Alert } from "react-bootstrap";
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

import { DEFAULT_MATURITIES, CHART_COLORS, MARKET_HOURS } from "../config";

/** ---------- Intraday time helpers (9:30 → 16:00 ET = 0..390 minutes) ---------- */

const SESSION_OPEN_MIN = 9 * 60 + 30; 
const SESSION_CLOSE_MIN = 16 * 60;    
const SESSION_LEN = SESSION_CLOSE_MIN - SESSION_OPEN_MIN; 

const toSessionMinutes = (isoOrDateLike) => {
  const d = new Date(isoOrDateLike);
  if (Number.isNaN(d.getTime())) return 0;
  const mins = d.getHours() * 60 + d.getMinutes();
  return Math.max(0, Math.min(SESSION_LEN, mins - SESSION_OPEN_MIN));
};

// Show labels at 10:00, 12:00, 14:00, 16:00 
const sessionTicks = [30, 150, 270, 390];
const START_MINUTE = 5; // drop 9:30–9:34; start plotting at 9:35

const minuteToLabel = (m) => {
  const total = SESSION_OPEN_MIN + m;
  const hh = Math.floor(total / 60);
  const mm = total % 60;
  if (mm !== 0) return ""; 
  const hr12 = hh % 12 === 0 ? 12 : hh % 12;
  return `${hr12} ${hh >= 12 ? "PM" : "AM"}`;
};

// Tooltip-friendly formatter for number axis using session minutes
const minuteToClock = (m) => {
  const total = SESSION_OPEN_MIN + m; // m is minutes from 9:30
  const hh = Math.floor(total / 60);
  const mm = total % 60;
  const hr12 = hh % 12 === 0 ? 12 : hh % 12;
  const ampm = hh >= 12 ? "PM" : "AM";
  return `${hr12}:${String(mm).padStart(2, "0")} ${ampm}`;
};
/** ----------------------------------------------------------------------------- */

// Custom tooltip for the multi-line chart
const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const yesterday = payload.find(p => p.dataKey === 'prev')?.value;
    const today = payload.find(p => p.dataKey === 'curr')?.value;
    return (
      <Card className="custom-tooltip shadow-sm small border-0">
        <Card.Body className="p-2">
          <p className="label mb-1"><strong>{minuteToClock(label)}</strong></p>
          {yesterday !== undefined && (
            <p className="intro mb-0" style={{ color: CHART_COLORS.yesterday }}>
              Previous: {yesterday.toFixed(3)}%
            </p>
          )}
          {today !== undefined && (
            <p className="intro mb-0" style={{ color: CHART_COLORS.today }}>
              Current: {today.toFixed(3)}%
            </p>
          )}
        </Card.Body>
      </Card>
    );
  }
  return null;
};

const pickAvailableMaturities = (raw, requested) =>
  requested.filter((m) => raw[m] && raw[m].all.length);

const normaliseYield = (raw) => {
  if (!Number.isFinite(raw)) return null;
  return raw > 50 ? raw / 100 : raw;
};

const isBeforeDataAvailabilityTime = () => {
  const now = new Date();
  const etNow = new Date(now.toLocaleString("en-US", { timeZone: "America/New_York" }));
  return (
    etNow.getHours() < MARKET_HOURS.dataAvailableHour ||
    (etNow.getHours() === MARKET_HOURS.dataAvailableHour &&
      etNow.getMinutes() < MARKET_HOURS.dataAvailableMinute)
  );
};

const buildSeries = (yesterdayArr, yesterdayDate, todayArr, todayDate, maturity) => {
  const combined = [];
  const push = (src, dateStr, segment) => {
    if (!src || !Array.isArray(src)) return;
    src.forEach((pt) => {
      const raw = parseFloat(pt[maturity]);
      const val = normaliseYield(raw);
      if (!Number.isFinite(val)) return;
      const ts = `${dateStr}T${pt.time}:00`;
      const minute = toSessionMinutes(ts);
      if (minute < START_MINUTE) return; // drop 9:30–9:34
      combined.push({
        ts,
        minute, // numeric x for fixed 4pm axis
        value: val,
        segment
      });
    });
  };
  push(yesterdayArr, yesterdayDate, "prev");
  if (todayArr && todayArr.length > 0) {
    if (!isBeforeDataAvailabilityTime()) {
      push(todayArr, todayDate, "curr");
    }
  }
  combined.sort((a, b) => new Date(a.ts) - new Date(b.ts));
  return {
    prev: combined.filter((d) => d.segment === "prev"),
    curr: combined.filter((d) => d.segment === "curr"),
    all: combined
  };
};

// Build Long-End (25–30Y) average series from per-maturity raw arrays
const buildLongEndAverageSeries = ({ yesterdayData, yesterdayDate, todayData, todayDate }) => {
  const MAT_KEYS = ['25', '26', '27', '28', '29', '30'];
  
  const combineFor = (arr, dateStr) => {
    if (!arr || !arr.length || !dateStr) return [];
    return arr.map((pt) => {
      const values = pt.values || pt.valueMap || pt;
      const nums = MAT_KEYS
        .map((k) => (values && values[k] != null ? Number(values[k]) : null))
        .filter((v) => typeof v === "number" && !Number.isNaN(v));
      const avg = nums.length ? nums.reduce((a, b) => a + b, 0) / nums.length : null;
      const timestamp = `${dateStr}T${pt.time}:00`;

      return {
        minute: toSessionMinutes(timestamp),
        value: normaliseYield(avg),
      };
    })
    // --- MODIFICATION START: Added check for minute >= START_MINUTE ---
    .filter((d) => d.value != null && d.minute >= START_MINUTE);
    // --- MODIFICATION END ---
  };

  const prev = combineFor(yesterdayData, yesterdayDate);
  const curr = combineFor(todayData, todayDate);

  // Combine both series into a single array for plotting
  const combined = prev.map(p => ({
    ...p,
    prev: p.value,
    curr: null,
  }));

  curr.forEach(c => {
    const existing = combined.find(item => item.minute === c.minute);
    if (existing) {
      existing.curr = c.value;
    } else {
      combined.push({
        minute: c.minute,
        prev: null,
        curr: c.value,
      });
    }
  });

  return combined.sort((a, b) => a.minute - b.minute);
};

// Compute domain and ticks at fixed 0.005 (0.5 bps) intervals
const computeDomainAndTicks = (data, valueKey = 'value') => {
  const fallback = { domain: ["dataMin", "dataMax"], ticks: undefined };
  if (!data || !data.length) return fallback;
  const values = data
    .map((item) => item[valueKey])
    .filter((val) => val !== undefined && val !== null && Number.isFinite(val));
  if (!values.length) return fallback;

  let minValue = Math.min(...values);
  let maxValue = Math.max(...values);

  const step = 0.005; // 0.5 basis points in % units

  const niceMin = Math.floor(minValue / step) * step;
  const niceMax = Math.ceil(maxValue / step) * step;

  const ticks = [];
  for (let t = niceMin; t <= niceMax + step / 2; t += step) {
    ticks.push(Number(t.toFixed(3)));
  }

  return { domain: [ticks[0], ticks[ticks.length - 1]], ticks };
};

function YieldCharts({ yieldData }) {
  const [series, setSeries] = useState({});
  const [maturities, setMaturities] = useState(DEFAULT_MATURITIES);
  const [loading, setLoading] = useState(true);
  const [yAxisDomains, setYAxisDomains] = useState({});
  const [yAxisTicks, setYAxisTicks] = useState({});
  const [showingOnlyYesterday, setShowingOnlyYesterday] = useState(false);
  const [yesterdayDate, setYesterdayDate] = useState("");
  const [todayDate, setTodayDate] = useState("");

  const [longEndSeries, setLongEndSeries] = useState([]);
  const [longEndTicks, setLongEndTicks] = useState([]);
  const [longEndDomain, setLongEndDomain] = useState([]);
  const [longEndPrevData, setLongEndPrevData] = useState([]);
  const [longEndCurrData, setLongEndCurrData] = useState([]);

  useEffect(() => {
    if (!yieldData || Object.keys(yieldData).length === 0) {
      setSeries({});
      setLongEndSeries([]);
      setLoading(false);
      return;
    }
    setLoading(true);
    try {
      const sortedDates = Object.keys(yieldData).sort();
      const yesterday =
        sortedDates[sortedDates.length > 1 ? sortedDates.length - 2 : sortedDates.length - 1];
      const today = sortedDates.length > 1 ? sortedDates[sortedDates.length - 1] : null;
      setYesterdayDate(yesterday);
      setTodayDate(today);

      const yesterdayData = yieldData[yesterday] || [];
      const todayData = today ? yieldData[today] || [] : [];
      setShowingOnlyYesterday(isBeforeDataAvailabilityTime() || !today || todayData.length === 0);

      const built = {};
      
      const LONG_END_MATURITIES = [25, 26, 27, 28, 29, 30];
      const individualMaturities = DEFAULT_MATURITIES.filter(
        (m) => m !== 'long_end' && !LONG_END_MATURITIES.includes(Number(m))
      );

      individualMaturities.forEach((m) => {
        built[m] = buildSeries(yesterdayData, yesterday, todayData, today, m);
      });
      
      const available = pickAvailableMaturities(built, individualMaturities);
      setMaturities(available);
      setSeries(built);

      const domains = {};
      const ticksByM = {};
      available.forEach((m) => {
        const { domain, ticks } = computeDomainAndTicks(built[m].all);
        domains[m] = domain;
        ticksByM[m] = ticks;
      });
      setYAxisDomains(domains);
      setYAxisTicks(ticksByM);

      // --- Long-end chart data processing ---
      const combinedLongEndData = buildLongEndAverageSeries({
        yesterdayData: yesterdayData,
        yesterdayDate: yesterday,
        todayData: todayData,
        todayDate: today,
      });
      setLongEndSeries(combinedLongEndData);

      const allLongEndValues = combinedLongEndData.flatMap(d => [d.prev, d.curr]).filter(v => v !== null);
      const { domain: longEndD, ticks: longEndT } = computeDomainAndTicks(
        allLongEndValues.map(v => ({ value: v }))
      );
      setLongEndDomain(longEndD);
      setLongEndTicks(longEndT);

      setLongEndPrevData(
        combinedLongEndData
          .filter(d => d.prev !== null)
          .map(d => ({ minute: d.minute, value: d.prev }))
      );
      setLongEndCurrData(
        combinedLongEndData
          .filter(d => d.curr !== null)
          .map(d => ({ minute: d.minute, value: d.curr }))
      );

    } catch (e) {
      console.error("Error processing yield data:", e);
      setSeries({});
      setLongEndSeries([]);
    } finally {
      setLoading(false);
    }
  }, [yieldData]);

  if (loading) {
    return (
      <div className="text-center p-5">
        <Spinner animation="border" />
      </div>
    );
  }
  if (!maturities.length && !longEndSeries.length) {
    return (
      <Alert variant="warning" className="m-3">
        No intraday yield-curve data available for the requested maturities.
      </Alert>
    );
  }

  const yesterdayLabel = yesterdayDate ? yesterdayDate.substring(5).replace("-", "/") : "";
  const todayLabel = todayDate ? todayDate.substring(5).replace("-", "/") : "";

  return (
    <div className="p-3">
      {showingOnlyYesterday && (
        <Alert variant="info" className="mb-3">
          On trading days, data will be available after 9:35 AM ET.
        </Alert>
      )}

      {/* Renders the charts for individual maturities, excluding the 25-30 year range */}
      {maturities.map((m) => (
        <Card key={m} className="shadow-sm mb-4">
          <Card.Header className="bg-light py-2">
            <h6 className="mb-0">{m}-Year Real-Time Yield (Total Investment Grade Market)</h6>
          </Card.Header>
          <Card.Body className="p-2">
            <Row className="g-1">
              {/* -------- Previous trading day (left) -------- */}
              <Col xs={12} md={6}>
                <h6 className="text-center mt-2">Previous Trading Day ({yesterdayLabel})</h6>
                <ResponsiveContainer width="100%" height={310}>
                  <LineChart
                    data={series[m].prev}
                    margin={{ top: 10, left: 50, right: 10, bottom: 10 }}
                  >
                    <CartesianGrid vertical={false} strokeDasharray="3 3" />
                    <XAxis
                      dataKey="minute"
                      type="number"
                      domain={[START_MINUTE, SESSION_LEN]}
                      ticks={sessionTicks}
                      tickFormatter={minuteToLabel}
                      interval={0}
                      axisLine
                      tickLine={false}
                      padding={{ left: 5, right: 20 }}
                    />
                    <YAxis
                      domain={yAxisDomains[m]}
                      ticks={yAxisTicks[m]}
                      tickFormatter={(v) => v.toFixed(3)}
                      axisLine={false}
                      tickLine={false}
                      orientation="left"
                      dx={-10}
                      allowDecimals
                      scale="linear"
                      width={40}
                      minTickGap={1}
                      tickMargin={8}
                    />
                    <Tooltip
                      labelFormatter={(mVal) => minuteToClock(mVal)}
                      formatter={(v) => `${v.toFixed(3)}%`}
                    />
                    <Line
                      type="monotone"
                      dataKey="value"
                      stroke={CHART_COLORS.yesterday}
                      strokeWidth={1.5}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Col>

              {/* -------- Current trading day (right, fixed to 4pm) -------- */}
              <Col xs={12} md={6}>
                <h6 className="text-center mt-2" style={{ color: CHART_COLORS.today }}>Current Trading Day ({todayLabel || " – "}) </h6>
                {showingOnlyYesterday || !series[m].curr?.length ? (
                  <div className="text-muted small p-3 text-center">Waiting for today's data…</div>
                ) : (
                  <ResponsiveContainer width="100%" height={310}>
                    <LineChart
                      data={series[m].curr}
                      margin={{ top: 10, left: 10, right: 50, bottom: 10 }}
                    >
                      <CartesianGrid vertical={false} strokeDasharray="3 3" />
                      <XAxis
                        dataKey="minute"
                        type="number"
                        domain={[START_MINUTE, SESSION_LEN + 10]}         // start at 9:35; keep 4 PM label safe
                        ticks={sessionTicks}
                        tickFormatter={minuteToLabel}
                        interval={0}
                        axisLine
                        tickLine={false}
                        padding={{ left: 20, right: 5 }}
                      />
                      <YAxis
                        domain={yAxisDomains[m]}
                        ticks={yAxisTicks[m]}
                        tickFormatter={(v) => v.toFixed(3)}
                        axisLine={false}
                        tickLine={false}
                        orientation="right"
                        dx={10}
                        allowDecimals
                        scale="linear"
                        width={40}
                        minTickGap={1}
                        tickMargin={8}
                      />
                      <Tooltip
                        labelFormatter={(mVal) => minuteToClock(mVal)}
                        formatter={(v) => `${v.toFixed(3)}%`}
                      />
                      <Line
                        type="monotone"
                        dataKey="value"
                        stroke={CHART_COLORS.today}
                        strokeWidth={1.5}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </Col>
            </Row>
          </Card.Body>
        </Card>
      ))}

      {/* Long-End (25–30Y) Average Yield Chart */}
      <Card className="shadow-sm mb-4">
        <Card.Header className="bg-light py-2">
          <h6 className="mb-0">Long-End (25–30Yr) Real-Time Yield (Total Investment Grade Market)</h6>
        </Card.Header>
        <Card.Body className="p-2">
          <Row className="g-1">
            {/* -------- Previous trading day (left) -------- */}
            <Col xs={12} md={6}>
              <h6 className="text-center mt-2">Previous Trading Day ({yesterdayLabel})</h6>
              <ResponsiveContainer width="100%" height={310}>
                <LineChart
                  data={longEndPrevData}
                  margin={{ top: 10, left: 50, right: 10, bottom: 10 }}
                >
                  <CartesianGrid vertical={false} strokeDasharray="3 3" />
                  <XAxis
                    dataKey="minute"
                    type="number"
                    domain={[START_MINUTE, SESSION_LEN]}
                    ticks={sessionTicks}
                    tickFormatter={minuteToLabel}
                    interval={0}
                    axisLine
                    tickLine={false}
                    padding={{ left: 5, right: 20 }}
                  />
                  <YAxis
                    domain={longEndDomain}
                    ticks={longEndTicks}
                    tickFormatter={(v) => v.toFixed(3)}
                    axisLine={false}
                    tickLine={false}
                    orientation="left"
                    dx={-10}
                    allowDecimals
                    scale="linear"
                    width={40}
                    minTickGap={1}
                    tickMargin={8}
                  />
                  <Tooltip
                    labelFormatter={(mVal) => minuteToClock(mVal)}
                    formatter={(v) => `${v.toFixed(3)}%`}
                  />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke={CHART_COLORS.yesterday}
                    strokeWidth={1.5}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </Col>

            {/* -------- Current trading day (right, fixed to 4pm) -------- */}
            <Col xs={12} md={6}>
              <h6 className="text-center mt-2" style={{ color: CHART_COLORS.today }}>Current Trading Day ({todayLabel || " – "}) </h6>
              {showingOnlyYesterday || !longEndCurrData.length ? (
                <div className="text-muted small p-3 text-center">Waiting for today's data…</div>
              ) : (
                <ResponsiveContainer width="100%" height={310}>
                  <LineChart
                    data={longEndCurrData}
                    margin={{ top: 10, left: 10, right: 50, bottom: 10 }}
                  >
                    <CartesianGrid vertical={false} strokeDasharray="3 3" />
                    <XAxis
                      dataKey="minute"
                      type="number"
                      domain={[START_MINUTE, SESSION_LEN + 10]}
                      ticks={sessionTicks}
                      tickFormatter={minuteToLabel}
                      interval={0}
                      axisLine
                      tickLine={false}
                      padding={{ left: 20, right: 5 }}
                    />
                    <YAxis
                      domain={longEndDomain}
                      ticks={longEndTicks}
                      tickFormatter={(v) => v.toFixed(3)}
                      axisLine={false}
                      tickLine={false}
                      orientation="right"
                      dx={10}
                      allowDecimals
                      scale="linear"
                      width={40}
                      minTickGap={1}
                      tickMargin={8}
                    />
                    <Tooltip
                      labelFormatter={(mVal) => minuteToClock(mVal)}
                      formatter={(v) => `${v.toFixed(3)}%`}
                    />
                    <Line
                      type="monotone"
                      dataKey="value"
                      stroke={CHART_COLORS.today}
                      strokeWidth={1.5}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </Col>
          </Row>
        </Card.Body>
      </Card>
    </div>
  );
}

export default YieldCharts;