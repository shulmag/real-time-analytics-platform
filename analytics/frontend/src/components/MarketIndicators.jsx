/*
 * @Date:
 */

import React, { useState } from "react";
import { Card, Row, Col, Alert, ButtonGroup, Button } from "react-bootstrap";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LabelList
} from "recharts";
import { CHART_COLORS } from "../config";

/*****************************************************************
 * HELPERS                                                       *
 *****************************************************************/

// comma‑separated integer
const fmtNumber = (n) => n.toLocaleString();
// billions from raw dollars
const fmtB = (n) => `${(n / 1000000000).toFixed(2)}B`;
// billions when value already expressed in millions
const fmtBfromM = (n) => `${(n / 1000).toFixed(2)}B`;

const formatYAxisValue = fmtNumber;
const formatValueCard  = (v) => (typeof v === "number" ? v.toFixed(2) : v ?? "-");
const formatChange     = (v) =>
  typeof v === "number" ? (v >= 0 ? `+${v.toFixed(2)}` : v.toFixed(2)) : "-";

/*****************************************************************
 * COMPONENT                                                     *
 *****************************************************************/

function MarketIndicators({ marketMetrics = {}, retailMetrics = {} }) {
  const [showComparison, setShowComparison] = useState(true); // default view: Current vs Previous Trading Day

  /* ------------------------------------------------------------
   * Build bar‑series (counts)
   * ---------------------------------------------------------- */
  const buildSeries = (m) => {
    const today = { 
      time: "Current Trading Day", 
      buys: m.buys ?? 0, 
      sells: m.sells ?? 0,
      buyVol: m.buyVol ?? 0,
      sellVol: m.sellVol ?? 0
    };
    
    const yday = {
      time: "Previous Trading Day",
      buys: m.yesterdayBuys ?? 0,
      sells: m.yesterdaySells ?? 0,
      buyVol: m.yesterdayBuyVol ?? 0,
      sellVol: m.yesterdaySellVol ?? 0
    };
    
    return showComparison ? [yday, today] : [today];
  };

  const seriesInst   = buildSeries(marketMetrics);
  const seriesRetail = buildSeries(retailMetrics);

  if (!seriesInst.length || !seriesRetail.length) {
    return (
      <Alert variant="warning" className="m-3">
        No market-indicator data available. Please check your API connection.
      </Alert>
    );
  }

  /* ------------------------------------------------------------ */
  const CustomTooltip = (isRetail) => ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;

    // Access the metrics object based on type
    const metrics = isRetail ? retailMetrics : marketMetrics;
    
    // Get volume based on which bar is being hovered (Current vs Previous Trading Day)
    let buyVolRaw, sellVolRaw;
    
    if (label === "Previous Trading Day") {
      // For previous day's tooltip, use yesterdayBuyVol and yesterdaySellVol
      buyVolRaw = metrics.yesterdayBuyVol ?? 0;
      sellVolRaw = metrics.yesterdaySellVol ?? 0;
    } else {
      // For current day, use buyVol and sellVol
      buyVolRaw = metrics.buyVol ?? 0;
      sellVolRaw = metrics.sellVol ?? 0;
    }

    // For retail, values are already in millions; for institutional they're in raw dollars
    const volBuy = isRetail ? fmtBfromM(buyVolRaw) : fmtB(buyVolRaw);
    const volSell = isRetail ? fmtBfromM(sellVolRaw) : fmtB(sellVolRaw);

    return (
      <div className="custom-tooltip bg-white p-2 border rounded shadow-sm">
        <p className="fw-bold mb-1">{label}</p>
        <p className="mb-0" style={{ color: CHART_COLORS.buys }}>
          Buys: {fmtNumber(payload.find((p) => p.dataKey === "buys").value)} | ${volBuy}
        </p>
        <p className="mb-0" style={{ color: CHART_COLORS.sells }}>
          Sells: {fmtNumber(payload.find((p) => p.dataKey === "sells").value)} | ${volSell}
        </p>
      </div>
    );
  };

  const renderCard = (title, series, metrics, isRetail) => (
    <Card className="shadow-sm h-100">
      <Card.Header className="bg-light"><h6 className="mb-0 fs-6">{title}</h6></Card.Header>
      <Card.Body>
        <h6 className="text-secondary">Daily trade count (buys vs sells)</h6>

        <ResponsiveContainer width="100%" height={280}>
          <BarChart 
            data={series}
            barSize={80}           // thinner bars (tweak 12–16 to taste)
            barCategoryGap="30%"   // more space between groups
            barGap={4}             // space between Buys and Sells bars
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            
            <YAxis tickFormatter={formatYAxisValue} />

            <Tooltip content={CustomTooltip(isRetail)} wrapperStyle={{ zIndex: 20 }} />
            <Legend />
            <Bar dataKey="buys" fill={CHART_COLORS.buys} name="Buys">
              <LabelList dataKey="buys" position="inside" formatter={fmtNumber} fill="white" />
            </Bar>
            <Bar dataKey="sells" fill={CHART_COLORS.sells} name="Sells">
              <LabelList dataKey="sells" position="inside" formatter={fmtNumber} fill="white" />
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        <Row className="mt-3">
          <Col md={12} className="mb-2">
            <Card className="bg-light h-100">
              <Card.Body className="py-2">
                <p className="mb-1 text-secondary small">Buy vs Sell Ratio</p>
                <h4 className="mb-0">{formatValueCard(metrics.buyVsSellRatio)}</h4>
                <small className={metrics.buyVsSellRatioChange >= 0 ? "text-success" : "text-danger"}>
                  {formatChange(metrics.buyVsSellRatioChange)}
                </small>
              </Card.Body>
            </Card>
          </Col>
        </Row>
      </Card.Body>
    </Card>
  );

  return (
    <div className="p-3">
      <div className="d-flex justify-content-between align-items-center mb-3">
        <h5 className="mb-0">Market Activity Indicators</h5>
        <ButtonGroup size="sm">
          <Button
            variant={!showComparison ? "primary" : "outline-primary"}
            onClick={() => setShowComparison(false)}
          >
            Current Trading Day Only
          </Button>
          <Button
            variant={showComparison ? "primary" : "outline-primary"}
            onClick={() => setShowComparison(true)}
          >
            Current vs Previous Trading Day
          </Button>
        </ButtonGroup>
      </div>

      <Row>
        <Col md={6} className="mb-4">
          {renderCard("Market Strength (> $5mm)", seriesInst, marketMetrics, false)}
        </Col>
        <Col md={6} className="mb-4">
          {renderCard("Retail Strength (< $1mm)", seriesRetail, retailMetrics, true)}
        </Col>
      </Row>
    </div>
  );
}

export default MarketIndicators;
