/**
 * Main Dashboard component for ficc analytics.
 * Displays yield curves, CoD values, and market indicators.
 */

import React, { useState, useEffect, forwardRef, useImperativeHandle } from 'react';
import { Card, Row, Col, Spinner, Alert, ProgressBar, Nav, Tab} from 'react-bootstrap';

// --- Long-end (25–30y) helpers ---
const LONG_END_KEYS = ['25','26','27','28','29','30'];

function averageLongEnd(valuesObj) {
  if (!valuesObj || typeof valuesObj !== 'object') return null;
  const keys = ['25','26','27','28','29','30'];
  const nums = keys.map(k => {
    const s = valuesObj?.[k];
    const n = valuesObj?.[Number(k)];
    const v = (typeof s === 'number' && Number.isFinite(s)) ? s
            : (typeof n === 'number' && Number.isFinite(n)) ? n
            : null;
    return v;
  }).filter(v => v != null);
  if (!nums.length) return null;
  return nums.reduce((a,b)=>a+b,0) / nums.length;
}
// NEW

import NProgress from 'nprogress';
import RealtimeYieldTable from './RealtimeYieldTable';
import 'nprogress/nprogress.css';
import RealtimeYieldCurve from './RealtimeYieldCurve';
// kill the built-in spinner (keep the slim top bar only)
NProgress.configure({ showSpinner: false });

import YieldCharts from './YieldCharts';
import MarketIndicators from './MarketIndicators';
import SpreadsChart from './SpreadsCharts';
import { apiService } from '../services/apiService';
import { DEFAULT_MATURITIES, DATE_FORMAT_OPTIONS } from '../config';
import moment from 'moment-timezone'
import MuniMarketStats from './MuniMarketStats';
import AAABenchmark from './AAABenchmark';

let dt = moment.tz('America/New_York').format('YYYY-MM-DD HH:mm')
const currentDateET = dt.substring(0,10)
const currentTimeET = dt.substring(11)

const Dashboard = forwardRef(({ onDataLoaded, onLoadingChange }, ref) => {
  const [pricingCurveData, setPricingCurveData] = useState(null);
  const [yesterdayPricingCurve, setYesterdayPricingCurve] = useState(null);
  const [pricingCurveDataTable, setPricingCurveDataTable] = useState(null);
  const [yesterdayPricingCurveTable, setYesterdayPricingCurveTable] = useState(null);
  const [loadingPricingTable, setLoadingPricingTable] = useState(true);
  const [loadingPricingCurve, setLoadingPricingCurve] = useState(true);

  const [loadingYieldCurves, setLoadingYieldCurves] = useState(true);
  const [loadingCodValues, setLoadingCodValues] = useState(true);
  const [loadingMarketMetrics, setLoadingMarketMetrics] = useState(true);
  const [loadingMuniStats, setLoadingMuniStats] = useState(true);

  //separate loading/error/data for 5Y and 10Y spreads
  const [loadingSpreads5Y, setLoadingSpreads5Y] = useState(true);
  const [loadingSpreads10Y, setLoadingSpreads10Y] = useState(true);
  const [errorSpreads5Y, setErrorSpreads5Y] = useState(null);
  const [errorSpreads10Y, setErrorSpreads10Y] = useState(null);
  const [spreadsData5Y_1mm, setSpreadsData5Y_1mm] = useState([]);
  const [spreadsData10Y_1mm, setSpreadsData10Y_1mm] = useState([]);
  const [loadingSpreads5Y_100, setLoadingSpreads5Y_100] = useState(true);
  const [loadingSpreads10Y_100, setLoadingSpreads10Y_100] = useState(true);
  const [errorSpreads5Y_100, setErrorSpreads5Y_100] = useState(null);
  const [errorSpreads10Y_100, setErrorSpreads10Y_100] = useState(null);
  const [spreadsData5Y_100, setSpreadsData5Y_100] = useState([]);
  const [spreadsData10Y_100, setSpreadsData10Y_100] = useState([]);

  const [activeTab, setActiveTab] = useState('overview'); // 'overview' or 'spreads'
  
  // 🔹 NEW: last good data snapshots (used to keep layout stable during refresh)
  const [lastYieldCurveData, setLastYieldCurveData] = useState({});
  const [lastCodValues, setLastCodValues] = useState({});
  const [lastMarketMetrics, setLastMarketMetrics] = useState({
    marketStrength: { monthly: [] },
    retailStrength: { monthly: [] }
  });
  const [lastPricingCurveData, setLastPricingCurveData] = useState(null);
  const [lastYesterdayPricingCurve, setLastYesterdayPricingCurve] = useState(null);
  const [lastPricingCurveDataTable, setLastPricingCurveDataTable] = useState(null);
  const [lastYesterdayPricingCurveTable, setLastYesterdayPricingCurveTable] = useState(null);

  // AAA Benchmark state (data + loading + last good snapshot)
  const [aaaData, setAaaData] = useState({});
  const [loadingAaa, setLoadingAaa] = useState(true);
  const [lastAaaData, setLastAaaData] = useState({});


  // 🔹 NEW: derived flags to prevent layout collapse during refresh
  const loadingSpreads =
    loadingSpreads5Y || loadingSpreads10Y || loadingSpreads5Y_100 || loadingSpreads10Y_100;

  const isActiveTabLoading =
    (activeTab === 'overview' && (loadingYieldCurves || loadingCodValues || loadingMarketMetrics)) ||
    (activeTab === 'spreads' && loadingSpreads) ||
    (activeTab === 'real-time-yield' && (loadingPricingCurve || loadingPricingTable)) ||
    (activeTab === 'aaaBenchmark' && loadingAaa) ||
    (activeTab === 'muniMarketStats' && loadingMuniStats);

  // Notify parent component when loading states change
  useEffect(() => {
    const loadingSpreadsLocal =   loadingSpreads5Y || loadingSpreads10Y || loadingSpreads5Y_100 || loadingSpreads10Y_100;
    if (onLoadingChange) {
      onLoadingChange({
        yieldCurves: loadingYieldCurves,
        codValues: loadingCodValues,
        marketMetrics: loadingMarketMetrics,
        spreads: loadingSpreadsLocal
      });
    }
  }, [
    loadingYieldCurves,
    loadingCodValues,
    loadingMarketMetrics,
    loadingSpreads5Y,
    loadingSpreads10Y,
    onLoadingChange
  ]);

  const [progress, setProgress] = useState(0);   // 0-100 for the bar
  const [error, setError] = useState(null);
  
  // State for yield curve data
  const [yieldCurveData, setYieldCurveData] = useState({});
  
  // State for CoD values
  const [codValues, setCodValues] = useState({});
  
  // State for market metrics
  const [marketMetrics, setMarketMetrics] = useState({
    marketStrength: { monthly: [] },
    retailStrength: { monthly: [] }
  });
  
  // State for current date/time information
  const [currentDate, setCurrentDate] = useState(
    new Date().toLocaleDateString('en-US', DATE_FORMAT_OPTIONS.short)
  );
  
  const [currentTime, setCurrentTime] = useState(
    new Date().toLocaleTimeString('en-US', DATE_FORMAT_OPTIONS.time)
  );

  // Format CoD values to 2 decimal places
  const formatCodValue = (value) => {
    if (typeof value === 'number') {
      return parseFloat(value.toFixed(3));
    }
    return value;
  };

  const formatCodChange = (change) => {
    if (typeof change === 'number') {
      return change; // Basis points, no decimals
    }
    return change;
  };

  // --- Long-end (25–30y) helpers --- // NEW
  const LONG_END_KEYS = ["25","26","27","28","29","30"];
  const averageLongEnd = (valuesObj) => { 
    const nums = LONG_END_KEYS
      .map(k => Number(valuesObj?.[k]))
      .filter(Number.isFinite);
    if (!nums.length) return null;
    return nums.reduce((a,b)=>a+b,0) / nums.length;
  };

  /**
   * Process yield curve data from API into format needed by charts
   */
  const processYieldCurveData = (apiData) => {
    if (!Array.isArray(apiData) || apiData.length === 0) {
      console.error("Invalid or empty yield curve data received:", apiData);
      return {};
    }

    const groupedByDate = {};

    apiData.forEach(point => {
      if (!point || !point.timestamp || !point.values) {
        console.warn("Skipping invalid data point:", point);
        return;
      }

      // Fix common "YYYY-MM-DD HH:mm[:ss]" -> "YYYY-MM-DDTHH:mm[:ss]" case
      let fixedTimestamp = point.timestamp;
      if (typeof point.timestamp === 'string' &&
          point.timestamp.includes(':') &&
          point.timestamp.indexOf(':') > 8) {
        const datePart = point.timestamp.substring(0, 10); // YYYY-MM-DD
        const timePart = point.timestamp.substring(11);
        const timeWithSeconds = timePart.split(':').length < 3 ? `${timePart}:00` : timePart;
        fixedTimestamp = `${datePart}T${timeWithSeconds}`;
      }

      const et = moment.tz(fixedTimestamp, 'America/New_York');
      if (!et.isValid()) {
        console.warn("Skipping invalid timestamp:", point.timestamp);
        return;
      }

      const dateStr = et.format('YYYY-MM-DD');   // key by ET date
      if (!groupedByDate[dateStr]) groupedByDate[dateStr] = [];

      groupedByDate[dateStr].push({
        time: et.format('HH:mm'),                // time string in ET
        long_end: averageLongEnd(point.values),  
        ...point.values
      });
    });

    return Object.keys(groupedByDate).reduce((result, date) => {
      result[date] = groupedByDate[date];
      return result;
    }, {});
  };
  
  /**
   * Generate Change-on-Day (CoD) values from yield curve data
   */
  const calculateCodValues = (yieldCurveData) => {
    // Need at least 1 day of data
    if (!yieldCurveData || Object.keys(yieldCurveData).length === 0) {
      console.warn("No yield curve data available for CoD calculation");
      return {};
    }
    
    // Get dates sorted chronologically 
    const dates = Object.keys(yieldCurveData).sort();
    if (dates.length === 0) return {};
    
    // Get the last available date for yesterday and today
    const yesterdayDate = dates[dates.length > 1 ? dates.length - 2 : 0];
    const todayDate = dates.length > 1 ? dates[dates.length - 1] : yesterdayDate;
    
    const yesterdayData = yieldCurveData[yesterdayDate] || [];
    const todayData = yieldCurveData[todayDate] || [];
    
    // Use the last data point of each day
    const yesterdayLast = yesterdayData.length > 0 ? yesterdayData[yesterdayData.length - 1] : null;
    const todayLast = todayData.length > 0 ? todayData[todayData.length - 1] : yesterdayLast;
    
    if (!yesterdayLast || !todayLast) {
      console.warn("Insufficient data for CoD calculation");
      return {};
    }
    
    // Calculate CoD values for selected maturities
    const codValues = {};
    const COD_MATS = [5, 10, 15, 20]; // NEW: restrict CoD to these

    COD_MATS.forEach(maturity => {
      const yesterdayValue = parseFloat(yesterdayLast[maturity.toString()]);
      const todayValue = parseFloat(todayLast[maturity.toString()]);
      
      if (!isNaN(yesterdayValue) && !isNaN(todayValue)) {
        // Normalize values (divide by 100 if needed)
        const normalizedYesterday = yesterdayValue > 50 ? yesterdayValue / 100 : yesterdayValue;
        const normalizedToday = todayValue > 50 ? todayValue / 100 : todayValue;
        
        // Calculate change in basis points
        const changeInBps = (normalizedToday - normalizedYesterday) * 100;
        
        codValues[maturity] = {
          yesterday: formatCodValue(normalizedYesterday),
          today: formatCodValue(normalizedToday),
          change: formatCodChange(changeInBps)
        };
      }
    });

    // NEW: Long-end CoD (25–30y average)
    const yLE = parseFloat(yesterdayLast['long_end']);
    const tLE = parseFloat(todayLast['long_end']);
    if (!isNaN(yLE) && !isNaN(tLE)) {
      const yN = yLE > 50 ? yLE / 100 : yLE;
      const tN = tLE > 50 ? tLE / 100 : tLE;
      codValues['long_end'] = {
        yesterday: formatCodValue(yN),
        today: formatCodValue(tN),
        change: formatCodChange((tN - yN) * 100)
      };
    }
    
    return codValues;
  };

  // Update the helper function
  const transformCodData = (codData) => {
    const out = {};
    Object.entries(codData).forEach(([m, d]) => {
      out[m] = {
        yesterday: formatCodValue(d.yesterday),
        today: formatCodValue(d.today),
        change: formatCodChange(d.change)
      };
    });
    return out;
  };

  /**
   * Fetch all data required for the dashboard
   */
  const fetchData = async (isRefresh = false) => {
    // Set all components to loading state
    setLoadingYieldCurves(true);
    setLoadingCodValues(true);
    setLoadingMarketMetrics(true);

    // reset spreads loading & errors
    setLoadingSpreads5Y(true);
    setLoadingSpreads10Y(true);
    setLoadingSpreads5Y_100(true);
    setLoadingSpreads10Y_100(true);

    setErrorSpreads5Y(null);
    setErrorSpreads10Y(null);
    setErrorSpreads5Y_100(null);
    setErrorSpreads10Y_100(null);

    setProgress(0);
    NProgress.start();
    setError(null);
    
    // Update time information
    const now = new Date();
    setCurrentDate(now.toLocaleDateString('en-US', DATE_FORMAT_OPTIONS.short));
    setCurrentTime(now.toLocaleTimeString('en-US', DATE_FORMAT_OPTIONS.time));
    
    // Use a simple approach: request the last 5 days of data
    const fiveDaysAgo = new Date(now);
    fiveDaysAgo.setDate(fiveDaysAgo.getDate() - 5);
    
    const requiredParams = {
      startDate: fiveDaysAgo.toISOString(),
      endDate: now.toISOString(),
      timeResolution: 'realtime'
    };

    const fetchPricingRealtimeYieldCurve = async () => {
      try {
        setLoadingPricingCurve(true);
        const todayCurve = await apiService.getRealtimeYieldCurvePlotFromPricing(currentDateET, currentTimeET);
        setPricingCurveData(todayCurve);
        setLastPricingCurveData(todayCurve); 

        const nowET = moment.tz('America/New_York');
        const yesterdayET = nowET.clone().subtract(1, 'day');
        const dateStr = yesterdayET.format('YYYY-MM-DD');
        const yesterdayCurve = await apiService.getRealtimeYieldCurvePlotFromPricing(dateStr, '16:00');
        setYesterdayPricingCurve(yesterdayCurve);
        setLastYesterdayPricingCurve(yesterdayCurve); 
      } catch (error) {
        console.error('Error fetching real-time curve Plot from pricing:', error);
      } finally {
        setLoadingPricingCurve(false);
      }
    };

    const fetchAaaBenchmark = async () => {
      setLoadingAaa(true);
      try {
        const json = await apiService.getAAABenchmark();
        const dataObj = json?.data ?? {};
        setAaaData(dataObj);
        setLastAaaData(dataObj); // keep a snapshot
        return true;
      } catch (err) {
        console.error('AAA benchmark fetch error:', err);
        setAaaData({});
        return false;
      } finally {
        setLoadingAaa(false);
      }
    };


    const fetchPricingRealtimeYieldCurveTable = async () => {
      try {
        setLoadingPricingTable(true);
        const todayCurve = await apiService.getRealtimeYieldCurveTableFromPricing(currentDateET, currentTimeET);
        setPricingCurveDataTable(todayCurve);
        setLastPricingCurveDataTable(todayCurve); 

        const nowET = moment.tz('America/New_York');
        const yesterdayET = nowET.clone().subtract(1, 'day');
        const dateStr = yesterdayET.format('YYYY-MM-DD');
        const yesterdayCurve = await apiService.getRealtimeYieldCurveTableFromPricing(dateStr, '16:00');
        setYesterdayPricingCurveTable(yesterdayCurve);
        setLastYesterdayPricingCurveTable(yesterdayCurve); 
      } catch (error) {
        console.error('Error fetching real-time curve Table from pricing:', error);
      } finally {
        setLoadingPricingTable(false);
      }
    };
    
    const fetchYieldCurves = async () => {
      try {
        setLoadingYieldCurves(true);
        const yieldCurveResponse = await apiService.getYieldCurves(
          requiredParams.timeResolution, 
          requiredParams.startDate, 
          requiredParams.endDate, 
          DEFAULT_MATURITIES,
          isRefresh
        );
        if (!yieldCurveResponse || !yieldCurveResponse.data) {
          throw new Error("Invalid response from yield curve API");
        }
        const processedData = processYieldCurveData(yieldCurveResponse.data);
        setYieldCurveData(processedData);
        setLastYieldCurveData(processedData); 
        setLoadingYieldCurves(false);

        const calculatedCodValues = calculateCodValues(processedData);
        setCodValues(calculatedCodValues);
        setLastCodValues(calculatedCodValues); 
        setLoadingCodValues(false);
        return true;
      } catch (err) {
        console.error("Error fetching yield curve data:", err);
        setLoadingYieldCurves(false);
        setLoadingCodValues(false);
        if (err.message && (
            err.message.includes('quota-exceeded') || 
            err.message.includes('insufficient resources'))) {
          throw new Error("Authentication rate limit exceeded. Please try again in a few minutes.");
        }
        throw err;
      }
    };
    
    const fetchMarketMetrics = async () => {
      try {
        setLoadingMarketMetrics(true);
        const metricsResponse = await apiService.getMarketMetrics(null, isRefresh);
        if (!metricsResponse || !metricsResponse.data) {
          throw new Error("Invalid response from market metrics API");
        }
        setMarketMetrics(metricsResponse.data);
        setLastMarketMetrics(metricsResponse.data); 
        setLoadingMarketMetrics(false);
        return true;
      } catch (err) {
        console.error("Error fetching market metrics:", err);
        setLoadingMarketMetrics(false);
        if (err.message && (
            err.message.includes('quota-exceeded') ||
            err.message.includes('insufficient resources'))) {
          throw new Error("Authentication rate limit exceeded. Please try again in a few minutes.");
        }
        throw err;
      }
    };
    
    // UPDATED: fetchSpreads now gets 4 datasets (2 quantity x 2 maturity)
    const fetchSpreads = async (isRefresh = false) => {
      // 1) reset flags
      setLoadingSpreads5Y(true);
      setLoadingSpreads10Y(true);
      setLoadingSpreads5Y_100(true);
      setLoadingSpreads10Y_100(true);

      setErrorSpreads5Y(null);
      setErrorSpreads10Y(null);
      setErrorSpreads5Y_100(null);
      setErrorSpreads10Y_100(null);

      // 2) request BOTH sizes: 1mm (=1000 bonds in your API) and 100 bonds
      const req5Y = apiService.getSpreads({
        maturities: ['4.5-5.5'],
        quantities: [1000, 100],
        days: 15,
        isRefresh
      });
      const req10Y = apiService.getSpreads({
        maturities: ['9.5-10.5'],
        quantities: [1000, 100],
        days: 15,
        isRefresh
      });

      const [r5, r10] = await Promise.allSettled([req5Y, req10Y]);

      // helper: split a payload by quantity no matter if it's a flat array or keyed
      const splitByQty = (payload, qty) => {
        if (!payload) return [];
        const raw = payload.data;
        if (Array.isArray(raw)) {
          return raw.filter(d => (d.quantity ?? qty) === qty);
        }
        if (raw && raw[qty]) return raw[qty];
        return [];
      };

      // 3) 5Y
      if (r5.status === 'fulfilled' && r5.value?.data) {
        const v = r5.value;
        const arr1mm = splitByQty(v, 1000);
        const arr100 = splitByQty(v, 100);

        setSpreadsData5Y_1mm(arr1mm);
        setSpreadsData5Y_100(arr100);
      } else {
        const msg = r5.status === 'rejected'
          ? (r5.reason?.message || String(r5.reason))
          : 'Invalid response for 5Y spreads';
        setErrorSpreads5Y(msg);
        setErrorSpreads5Y_100(msg);
        setSpreadsData5Y_1mm([]);
        setSpreadsData5Y_100([]);
      }

      // 4) 10Y
      if (r10.status === 'fulfilled' && r10.value?.data) {
        const v = r10.value;
        const arr1mm = splitByQty(v, 1000);
        const arr100 = splitByQty(v, 100);

        setSpreadsData10Y_1mm(arr1mm);
        setSpreadsData10Y_100(arr100);
      } else {
        const msg = r10.status === 'rejected'
          ? (r10.reason?.message || String(r10.reason))
          : 'Invalid response for 10Y spreads';
        setErrorSpreads10Y(msg);
        setErrorSpreads10Y_100(msg);
        setSpreadsData10Y_1mm([]);
        setSpreadsData10Y_100([]);
      }

      // 5) clear loading flags
      setLoadingSpreads5Y(false);
      setLoadingSpreads10Y(false);
      setLoadingSpreads5Y_100(false);
      setLoadingSpreads10Y_100(false);

      return true;
    };

    
    try {
      // Run requests in parallel
      const results = await Promise.allSettled([
        fetchYieldCurves(),
        fetchMarketMetrics(),
        fetchSpreads(isRefresh),
        fetchPricingRealtimeYieldCurve(),
        fetchPricingRealtimeYieldCurveTable(),
        fetchAaaBenchmark()
      ]);
      
      const errors = results
        .filter(result => result.status === 'rejected')
        .map(result => result.reason);
      
      if (errors.length > 0) {
        if (errors.some(err => 
            err.message && (
              err.message.includes('quota-exceeded') || 
              err.message.includes('authentication rate limit')))) {
          setError("Firebase authentication rate limit exceeded. Please try again in a few minutes.");
        } else {
          setError(errors[0].message || 'An error occurred while fetching data');
        }
      }
      
      setProgress(100);
      NProgress.set(1.0);
      
      if (onDataLoaded) {
        onDataLoaded();
      }
    } catch (err) {
      setError(err.message || 'An error occurred while fetching data');
      console.error('Error fetching data:', err);
    } finally {
      NProgress.done();
    }
  };

  // Expose the refreshData method through the ref
  useImperativeHandle(ref, () => ({
      refreshData: () => fetchData(), // return the Promise
  }));

  // Fetch data on component mount with retry and throttling
  useEffect(() => {
    let retryCount = 0;
    const maxRetries = 3;
    let retryTimeout = null;
    
    const attemptFetch = () => {
      fetchData()
        .catch(error => {
          console.error(`Data fetch attempt ${retryCount + 1} failed:`, error);
          if (retryCount < maxRetries) {
            retryCount++;
            const delay = Math.pow(2, retryCount) * 1000;
            console.log(`Retrying in ${delay}ms...`);
            if (retryTimeout) {
              clearTimeout(retryTimeout);
            }
            retryTimeout = setTimeout(attemptFetch, delay);
          } else {
            setError(`Failed to fetch data after ${maxRetries} attempts. Please try again later or contact support.`);
          }
        });
    };
    
    attemptFetch();
    return () => {
      if (retryTimeout) {
        clearTimeout(retryTimeout);
      }
    };
  }, []);

  // 🔹 Display data picks current if ready, else last good snapshot
  const displayYieldCurveData = Object.keys(yieldCurveData).length ? yieldCurveData : lastYieldCurveData;
  const displayCodValues = Object.keys(codValues).length ? codValues : lastCodValues;
  const displayMarketMetrics = (marketMetrics?.marketStrength?.monthly?.length || marketMetrics?.retailStrength?.monthly?.length)
    ? marketMetrics
    : lastMarketMetrics;

  const displayPricingCurveData = pricingCurveData || lastPricingCurveData;
  const displayYesterdayPricingCurve = yesterdayPricingCurve || lastYesterdayPricingCurve;
  const displayPricingCurveDataTable = pricingCurveDataTable || lastPricingCurveDataTable;
  const displayYesterdayPricingCurveTable = yesterdayPricingCurveTable || lastYesterdayPricingCurveTable;
  const displayAaaData = Object.keys(aaaData || {}).length ? aaaData : lastAaaData;

  return (
    <div className="dashboard">
      {/* Error state */}
      {error && (
        <Alert variant="danger" className="m-3">
          {error}
        </Alert>
      )}
      
      {/* Tab navigation */}
      {!error && (
        <Tab.Container
          activeKey={activeTab}
          onSelect={setActiveTab}
          mountOnEnter
          unmountOnExit
        >
          <Nav variant="tabs" className="px-3 pt-3 bg-white border-bottom">
            <Nav.Item>
              <Nav.Link eventKey="overview">Muni Market Overview</Nav.Link>
            </Nav.Item>
            <Nav.Item>
              <Nav.Link eventKey="real-time-yield">Real Time Yield Curve (IG)</Nav.Link>
            </Nav.Item>
            <Nav.Item>
              <Nav.Link eventKey="aaaBenchmark">Real Time AAA Yield</Nav.Link>
            </Nav.Item>
            <Nav.Item>
              <Nav.Link eventKey="spreads">Dollar Price Spreads</Nav.Link>
            </Nav.Item>
              <Nav.Item>
              <Nav.Link eventKey="muniMarketStats">Real Time Trade Statistics</Nav.Link>
            </Nav.Item>
          </Nav>
          
          <Tab.Content>
            {/* Wrapper: prevents layout collapse + shows overlay while active tab is loading */}
            <div style={{ position: 'relative', minHeight: '60vh' }}>
              {isActiveTabLoading && (
                <div
                  style={{
                    position: 'absolute',
                    inset: 0,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    background: 'rgba(255,255,255,0.6)',
                    zIndex: 10,
                  }}
                >
                  <Spinner animation="border" />
                </div>
              )}

              {/* Overview Tab */}
              <Tab.Pane eventKey="overview" mountOnEnter unmountOnExit>
                {/* CoD Cards — render using display snapshot even while loading */}
                {Object.keys(displayCodValues).length > 0 && (
                  <Row className="mx-0 mt-3 p-2 bg-light">
                    {Object.entries(displayCodValues).map(([maturity, data]) => (
                      <Col key={maturity} className="mb-2 cod-card-5-col">
                        <Card className="h-100 shadow-sm">
                          <Card.Body className="p-2">
                            <h6 className="card-title mb-2">
                              {maturity === 'long_end' ? 'Long End CoD' : `${maturity}-yr CoD`}{/* NEW */}
                            </h6>
                            <p className="mb-1 small text-muted">
                              Previous Close&nbsp;(4 PM): <strong>{data.yesterday.toFixed(2)}%</strong>
                            </p>
                            <p className="mb-1 small text-muted">
                              Current&nbsp;(real-time): <strong>{data.today.toFixed(2)}%</strong>
                            </p>
                            <h4 className={`mb-0 ${Math.round((data.today.toFixed(2) - data.yesterday.toFixed(2))*100) > 0 ? 'text-danger' : 'text-success'}`}>
                              {Math.round((data.today.toFixed(2) - data.yesterday.toFixed(2))*100) > 0  ? '+' : ''}{Math.round((data.today.toFixed(2) - data.yesterday.toFixed(2))*100)} bps
                            </h4>
                          </Card.Body>
                        </Card>
                      </Col>
                    ))}
                  </Row>
                )}
                
                {/* Yield Charts */}
                {Object.keys(displayYieldCurveData).length > 0 && (
                  <div className="mb-2">
                    <YieldCharts yieldData={displayYieldCurveData} />
                  </div>
                )}
                
                {/* Market Indicators */}
                {displayMarketMetrics && displayMarketMetrics.marketStrength && (
                  <div>
                    <MarketIndicators 
                      marketStrengthData={displayMarketMetrics.marketStrength.monthly}
                      retailStrengthData={displayMarketMetrics.retailStrength.monthly}
                      marketMetrics={displayMarketMetrics.marketStrength}
                      retailMetrics={displayMarketMetrics.retailStrength}
                    />
                  </div>
                )}
              </Tab.Pane>
              
              {/* Spreads Tab */}
              <Tab.Pane eventKey="spreads" mountOnEnter unmountOnExit>
                <SpreadsChart 
                  // data
                  spreadsData5Y_1mm={spreadsData5Y_1mm}
                  spreadsData10Y_1mm={spreadsData10Y_1mm}
                  spreadsData5Y_100={spreadsData5Y_100}
                  spreadsData10Y_100={spreadsData10Y_100}

                  // loading per series
                  loading5Y_1mm={loadingSpreads5Y}
                  loading10Y_1mm={loadingSpreads10Y}
                  loading5Y_100={loadingSpreads5Y_100}
                  loading10Y_100={loadingSpreads10Y_100}

                  // errors per series
                  error5Y_1mm={errorSpreads5Y}
                  error10Y_1mm={errorSpreads10Y}
                  error5Y_100={errorSpreads5Y_100}
                  error10Y_100={errorSpreads10Y_100}
                />
              </Tab.Pane>

              {/* Real Time Yield Curve Plot (IG) */}
              <Tab.Pane eventKey="real-time-yield" mountOnEnter unmountOnExit>
                {/* Use display snapshots so layout remains while refreshing */}
                {displayPricingCurveData && displayPricingCurveDataTable && displayYesterdayPricingCurveTable && (
                <>
                  <div className="my-3">
                    <RealtimeYieldCurve 
                      yield_data={displayPricingCurveData} 
                      yesterday_data={displayYesterdayPricingCurve} 
                    />
                  </div>
                  <div className="my-3">
                    <RealtimeYieldTable 
                      todayData={displayPricingCurveDataTable} 
                      yesterdayData={displayYesterdayPricingCurveTable} 
                      dateToday={currentDateET} 
                    />
                  </div>         
                </>       
                )}
              </Tab.Pane>
              <Tab.Pane eventKey="aaaBenchmark" mountOnEnter unmountOnExit>
                {Object.keys(displayAaaData || {}).length > 0 ? (
                  <div className="mb-2">
                    <AAABenchmark yieldData={displayAaaData} />
                  </div>
                ) : (
                  <div className="text-muted small p-3 text-center">
                    Loading AAA Benchmark…
                  </div>
                )}
              </Tab.Pane>
              <Tab.Pane eventKey="muniMarketStats" mountOnEnter unmountOnExit>
                <MuniMarketStats onLoadingChange={setLoadingMuniStats} />
              </Tab.Pane>
            </div>
          </Tab.Content>
        </Tab.Container>
      )}
    </div>
  );
});

export default Dashboard;