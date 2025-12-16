/*
 * @Date: 2025-09-11
 */

/**
 * YieldCharts component for displaying AAA Benchmarks.
 * Yesterday's chart has Y-axis on the left, today's chart has Y-axis on the right.
 * Margins tuned to avoid clipping of the 4 PM label, Y-axis labels moved outward, and charts positioned closer together.
 * Y-axis ticks are computed at fixed 0.5 basis point intervals (0.005%) per chart.
 * Plot window is 9:35–4:00 PM ET (minutes 5–390). Points before 9:35 are dropped.
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
  Legend,
} from "recharts";

import { DEFAULT_MATURITIES, CHART_COLORS, MARKET_HOURS } from "../config";

/** ---------- Intraday time helpers (9:30 → 16:00 ET = 0..390 minutes) ---------- */

const SESSION_OPEN_MIN = 9 * 60 + 30;
const SESSION_CLOSE_MIN = 16 * 60;
const SESSION_LEN = SESSION_CLOSE_MIN - SESSION_OPEN_MIN;

const toSessionMinutes = (hhmm) => {
  if (typeof hhmm !== "string" || !hhmm.includes(":")) return 0;
  const [h, m] = hhmm.split(":").map((n) => parseInt(n, 10));
  const mins = h * 60 + m; // clock minutes in ET
  return Math.max(0, Math.min(SESSION_LEN, mins - SESSION_OPEN_MIN));
};

// Show labels at 10:00, 12:00, 14:00, 16:00
const sessionTicks = [30, 150, 270, 390];
const START_MINUTE = 1;

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

const pickAvailableMaturities = (raw, requested) =>
  requested.filter((m) => raw[m] && raw[m].all.length);

const normaliseYield = (raw) => {
  if (!Number.isFinite(raw)) return null;
  return raw > 50 ? raw / 100 : raw;
};

const isBeforeDataAvailabilityTime = () => {
  const now = new Date();
  const etNow = new Date(
    now.toLocaleString("en-US", { timeZone: "America/New_York" })
  );
  return (
    etNow.getHours() < MARKET_HOURS.dataAvailableHour ||
    (etNow.getHours() === MARKET_HOURS.dataAvailableHour &&
      etNow.getMinutes() < MARKET_HOURS.dataAvailableMinute)
  );
};

// --- NEW: Function to calculate Change-on-Day values ---
const calculateCodValues = (data, maturities) => {
  if (!data || Object.keys(data).length === 0) return {};

  const dates = Object.keys(data).sort();
  const yesterdayDate =
    dates.length > 1 ? dates[dates.length - 2] : dates[0];
  const todayDate = dates.length > 1 ? dates[dates.length - 1] : yesterdayDate;

  const yesterdayData = data[yesterdayDate] || [];
  const todayData = data[todayDate] || [];

  // Use the last data point of each day
  const yesterdayLast =
    yesterdayData.length > 0
      ? yesterdayData[yesterdayData.length - 1]
      : null;
  const todayLast =
    todayData.length > 0 ? todayData[todayData.length - 1] : yesterdayLast;

  if (!yesterdayLast || !todayLast) return {};

  const cods = {};
  maturities.forEach((maturity) => {
    const yesterdayValue = parseFloat(yesterdayLast[maturity]);
    const todayValue = parseFloat(todayLast[maturity]);

    if (!isNaN(yesterdayValue) && !isNaN(todayValue)) {
      const normalizedYesterday = normaliseYield(yesterdayValue);
      const normalizedToday = normaliseYield(todayValue);

      // This is now just for data; rendering logic will be separate
      const changeInBps = (normalizedToday - normalizedYesterday) * 100;

      cods[maturity] = {
        yesterday: normalizedYesterday,
        today: normalizedToday,
        change: changeInBps,
      };
    }
  });
  return cods;
};

const buildSeries = (
  yesterdayArr,
  yesterdayDate,
  todayArr,
  todayDate,
  maturity
) => {
  const combined = [];
  const push = (src, dateStr, segment) => {
    if (!src || !Array.isArray(src)) return;
    src.forEach((pt) => {
      const raw = parseFloat(pt[maturity]);
      const val = normaliseYield(raw);
      if (!Number.isFinite(val)) return;

      const minute = toSessionMinutes(pt.time);
      if (minute < START_MINUTE) return; // drop 9:30–9:34

      combined.push({
        ts: `${dateStr}T${pt.time}:00`,
        minute,
        value: val,
        segment,
      });
    });
  };
  push(yesterdayArr, yesterdayDate, "prev");
  if (todayArr && todayArr.length > 0) {
    if (!isBeforeDataAvailabilityTime()) {
      push(todayArr, todayDate, "curr");
    }
  }
  combined.sort((a, b) => a.minute - b.minute);

  return {
    prev: combined.filter((d) => d.segment === "prev"),
    curr: combined.filter((d) => d.segment === "curr"),
    all: combined,
  };
};

// Compute domain and ticks at fixed 0.005 (0.5 bps) intervals
const computeDomainAndTicks = (data) => {
  const fallback = { domain: ["dataMin", "dataMax"], ticks: undefined };
  if (!data || !data.length) return fallback;
  const values = data
    .map((item) => item.value)
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

function AAABenchmark({ yieldData }) {
  const [series, setSeries] = useState({});
  const [maturities, setMaturities] = useState(DEFAULT_MATURITIES);
  const [loading, setLoading] = useState(true);
  const [yAxisDomains, setYAxisDomains] = useState({});
  const [yAxisTicks, setYAxisTicks] = useState({});
  const [showingOnlyYesterday, setShowingOnlyYesterday] = useState(false);
  const [yesterdayDate, setYesterdayDate] = useState("");
  const [todayDate, setTodayDate] = useState("");
  const [codValues, setCodValues] = useState({});

  useEffect(() => {
    if (!yieldData || Object.keys(yieldData).length === 0) {
      setSeries({});
      setCodValues({});
      setLoading(false);
      return;
    }
    setLoading(true);
    try {
      const sortedDates = Object.keys(yieldData).sort();
      const yesterday =
        sortedDates[sortedDates.length > 1
            ? sortedDates.length - 2
            : sortedDates.length - 1];
      const today =
        sortedDates.length > 1 ? sortedDates[sortedDates.length - 1] : null;
      setYesterdayDate(yesterday);
      setTodayDate(today);

      const yesterdayData = yieldData[yesterday] || [];
      const todayData = today ? yieldData[today] || [] : [];
      setShowingOnlyYesterday(
        isBeforeDataAvailabilityTime() || !today || todayData.length === 0
      );

      const built = {};
      DEFAULT_MATURITIES.forEach((m) => {
        built[m] = buildSeries(yesterdayData, yesterday, todayData, today, m);
      });
      const available = pickAvailableMaturities(built, DEFAULT_MATURITIES);
      setMaturities(available);
      setSeries(built);

      const calculatedCodValues = calculateCodValues(yieldData, available);
      setCodValues(calculatedCodValues);

      const domains = {};
      const ticksByM = {};
      available.forEach((m) => {
        const { domain, ticks } = computeDomainAndTicks(built[m].all);
        domains[m] = domain;
        ticksByM[m] = ticks;
      });
      setYAxisDomains(domains);
      setYAxisTicks(ticksByM);
    } catch (e) {
      console.error("Error processing yield data:", e);
      setSeries({});
      setCodValues({});
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
  if (!maturities.length) {
    return (
      <Alert variant="warning" className="m-3">
        No intraday yield-curve data available for the requested maturities.
      </Alert>
    );
  }

  const yesterdayLabel = yesterdayDate
    ? yesterdayDate.substring(5).replace("-", "/")
    : "";
  const todayLabel = todayDate ? todayDate.substring(5).replace("-", "/") : "";

  return (
    <div>
      {/* --- MODIFICATION START: Updated CoD card styling to perfectly match Dashboard --- */}
      {Object.keys(codValues).length > 0 && (
        <Row className="mx-0 mt-3 p-2 bg-light">
          {Object.entries(codValues).map(([maturity, data]) => (
            <Col key={maturity} className="mb-2 cod-card-5-col">
              <Card className="h-100 shadow-sm">
                <Card.Body className="p-2">
                  <h6 className="card-title mb-2">{maturity}-yr CoD</h6>
                  <p className="mb-1 small text-muted">
                    Previous Close&nbsp;(4 PM):{" "}
                    <strong>{data.yesterday.toFixed(2)}%</strong>
                  </p>
                  <p className="mb-1 small text-muted">
                    Current&nbsp;(real-time):{" "}
                    <strong>{data.today.toFixed(2)}%</strong>
                  </p>
                  <h4
                    className={`mb-0 ${
                      Math.round(
                        (data.today.toFixed(2) - data.yesterday.toFixed(2)) * 100
                      ) > 0
                        ? "text-danger"
                        : "text-success"
                    }`}
                  >
                    {Math.round(
                      (data.today.toFixed(2) - data.yesterday.toFixed(2)) * 100
                    ) > 0
                      ? "+"
                      : ""}
                    {Math.round(
                      (data.today.toFixed(2) - data.yesterday.toFixed(2)) * 100
                    )}{" "}
                    bps
                  </h4>
                </Card.Body>
              </Card>
            </Col>
          ))}
        </Row>
      )}
      {/* --- MODIFICATION END --- */}
    <div className="p-3">
      {showingOnlyYesterday && (
        <Alert variant="info" className="mb-3">
          On trading days, data will be available after 9:35 AM ET.
        </Alert>
      )}
      
      {maturities.map((m) => (
        <Card key={m} className="shadow-sm mb-4">
          <Card.Header className="bg-light py-2">
            <h6 className="mb-0">{m}-Year Real-Time Yield (AAA)</h6>
          </Card.Header>
          <Card.Body className="p-2">
            <Row className="g-1">
              {/* -------- Previous trading day (left) -------- */}
              <Col xs={12} md={6}>
                <h6 className="text-center mt-2">
                  Previous Trading Day ({yesterdayLabel})
                </h6>
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
                <h6
                  className="text-center mb-2"
                  style={{ color: CHART_COLORS.today }}
                >
                  Current Trading Day ({todayLabel || " – "}){" "}
                </h6>
                {showingOnlyYesterday || !series[m].curr?.length ? (
                  <div className="text-muted small p-3 text-center">
                    Waiting for today's data…
                  </div>
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
                        domain={[START_MINUTE, SESSION_LEN + 10]} // start at 9:35; keep 4 PM label safe
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
    </div>
    </div>
  );
}

export default AAABenchmark;