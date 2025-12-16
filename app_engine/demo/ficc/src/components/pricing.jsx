/*
 * @Date: 2022-11-10
 */

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { getAuth, onAuthStateChanged } from 'firebase/auth';
import moment from 'moment-timezone';
import { useTable, usePagination, useSortBy } from 'react-table';
// import { FaSort, FaSortUp, FaSortDown } from 'react-icons/fa';
import { Tooltip } from 'react-tooltip';
import { Spinner } from 'react-bootstrap';

// React Bootstrap components
import Container from 'react-bootstrap/Container';
import Card from 'react-bootstrap/Card';
import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';

// Custom components in ./pricing/
import NavBarTop from './navBarTop.jsx';
import TabsSearchForm from './pricing/tabsCusipSearchForm.jsx';
import ExpandableTradeTable from './pricing/ExpandableTradeTable.jsx';
import PricingResults from './pricing/PricingResults.jsx';
import SimilarBonds from './pricing/similarBonds.jsx';
import BatchPricing from './pricing/BatchPricing.jsx';
import { statesDict, purposeClassDict, ratingsDict } from './pricing/relatedVarDict.js';

// Services and utilities
import { getPrice, uploadFile, getSimilarBonds } from '../services/priceService';

import FONT_SIZE from './pricing/globalVariables';

// Styles
import '../styles/ficc-tables.css';

function Pricing() {
  let dt = moment.tz('America/New_York').format('YYYY-MM-DD HH:mm');
  const currentDate = dt.substring(0, 10);
  const currentTime = dt.substring(11);

  const loggedOutMessage = 'You have been logged out due to a period of inactivity. Refresh the page!';
  const nav = useNavigate();

  // ----------------------------------------
  //  Authentication & Lifecycle
  // ----------------------------------------
  const [stateToken, setStateToken] = useState('');
  const [userEmail, setUserEmail] = useState('');
  useEffect(() => {
    const auth = getAuth();
    
    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      if (user) {
        try {
          const token = await user.getIdToken(true);
          setStateToken(token);
          setUserEmail(user.email);
        } catch (error) {
          console.error('Initial token fetch error:', error);
          if (error.code && error.code.includes('auth/')) {
            redirectToLogin();
          }
        }
      } else {
        redirectToLogin();
      }
    });

    // Cleanup subscription
    return () => unsubscribe();
  }, []);

  function redirectToLogin() {
    nav('/login');
  }

  // ----------------------------------------
  //  Single Pricing tab state
  // ----------------------------------------
  const [loadingMessage, setLoadingMessage] = useState();
  const [key, setKey] = useState('pricing');
  const [predictedPrice, setPredictedPrice] = useState('');
  const [calcMethod, setCalcMethod] = useState('');
  const [ytw, setYtw] = useState('');
  const [usedDollarPriceModel, setUsedDollarPriceModel] = useState(false);
  const [reasonForUsingDollarPriceModel, setReasonForUsingDollarPriceModel] = useState('');
  const [open, setOpen] = useState(false);
  const [isPriceHidden, setIsPriceHidden] = useState(true);
  const [isRelatedLoading, setIsRelatedLoading] = useState(false);
  const [noRelatedBondsFound, setNoRelatedBondsFound] = useState(true);
  const [similarBondsSearchHasRun, setSimilarBondsSearchHasRun] = useState(false);
  const [isFirstTime, setIsFirstTime] = useState(true);
  const [isPricing, setIsPricing] = useState(false);
  const [similarBondsRes, setSimilarBondsRes] = useState([]);
  const [tradeHistory, setTradeHistory] = useState([]);
  const defaultCusip = '13063D7Q5';

  // Similar bonds config
  const defaultDesc = '';
  const defaultMinCoupon = 0;
  const defaultMaxCoupon = 1000;
  const defaultMinMaturityDate = '2025-01-01';
  const defaultMaxMaturityDate = '2125-12-31';
  const defaultRelatedSearchVal = {
    desc: defaultDesc,
    minCoupon: defaultMinCoupon,
    maxCoupon: defaultMaxCoupon,
    minMaturityDate: defaultMinMaturityDate,
    maxMaturityDate: defaultMaxMaturityDate,
    radio: 'previous_day',
    issuerChoice: 'any_issuer'
  };
  const [relatedSearchVal, setRelatedSearchVal] = useState(defaultRelatedSearchVal);

  const [referenceFeatures, setReferenceFeatures] = useState({});

  // Single pricing form defaults
  const defaultQuantity = 500;
  const defaultTradeType = 'S';
  const [cusipForDisplay, setCusipForDisplay] = useState(defaultCusip);
  const [displayTextForYtw, setDisplayTextForYtw] = useState('Worst');
  const [displayPriceForYtw, setDisplayPriceForYtw] = useState(100);

  const [searchValues, setSearchValues] = useState({
    cusip: defaultCusip,
    amount: defaultQuantity,
    tradeType: defaultTradeType,
    date: currentDate,
    time: currentTime,
    token: ''
  });

  // ----------------------------------------
  //  Batch Pricing states
  // ----------------------------------------
  const [file, setFile] = useState();
  const [batchValues, setBatchValues] = useState({
    quantity: defaultQuantity,
    tradeType: defaultTradeType
  });
  const [isBatchProcessing, setIsBatchProcessing] = useState(false);
  const [isDownloadProcessing, setIsDownloadProcessing] = useState(false);
  const [tableData, setTableData] = useState([]);
  const [showTable, setShowTable] = useState(false);

  // ----------------------------------------
  //  React-Table Setup (batch results)
  // ----------------------------------------
  const resultsPerPage = 25;
  const columns = React.useMemo(
    () => [
      { Header: 'CUSIP', accessor: 'cusip' },
      {
        Header: 'Quantity',
        accessor: 'quantity',
        Cell: ({ value }) => value.toLocaleString()
      },
      { Header: 'Trade Type', accessor: 'trade_type' },
      { Header: 'YTW', accessor: 'ytw' },
      { Header: 'Price', accessor: 'price' },
      { Header: 'Priced to date', accessor: 'yield_to_worst_date' },
      { Header: 'Coupon', accessor: 'coupon' },
      { Header: 'Security Description', accessor: 'security_description' },
      { Header: 'Maturity Date', accessor: 'maturity_date' },
      { Header: 'Notes', accessor: 'error_message' }
    ],
    []
  );

  const {
    getTableProps,
    getTableBodyProps,
    headerGroups,
    page,
    prepareRow,
    canPreviousPage,
    canNextPage,
    pageOptions,
    pageCount,
    gotoPage,
    nextPage,
    previousPage,
    setPageSize,
    state: { pageIndex, pageSize }
  } = useTable(
    {
      columns,
      data: tableData,
      initialState: { pageIndex: 0, pageSize: resultsPerPage }
    },
    useSortBy,
    usePagination
  );

  // ----------------------------------------
  //  Options
  // ----------------------------------------
  const tradeType = [
    { key: 'D', text: 'Inter-Dealer' },
    { key: 'P', text: 'Bid Side' },
    { key: 'S', text: 'Offered Side' }
  ];

  const dollarPriceModelDisplayText = {
    missing_or_negative_yields: [
      'Missing or negative yields reported',
      'We do not provide an evaluated yield since previous MSRB reported yields for this CUSIP are missing or negative.'
    ],
    adjustable_rate_coupon: [
      'Adjustable rate coupon',
      'For adjustable rate coupon, we do not yet display yield. Yield to conversion date coming soon!'
    ], 
    maturing_soon: [
      'Maturing soon',
      'CUSIP is maturing very soon or has already matured so we only provide a dollar price.'
    ], 
    defaulted: [
      'CUSIP has defaulted', 
      'CUSIP has defaulted so we only provide a dollar price.'
    ], 
    high_yield_in_history: [
      'Abnormally high (greater than 10%) yield in history',
      'MSRB reported yields for this CUSIP are abnormally high (greater than 10%), so we only provide a dollar price.'
    ]
  };

  // ----------------------------------------
  //  Handlers
  // ----------------------------------------
  // Single pricing form updates
  function set(name) {
    return function ({ target: { value } }) {
      setSearchValues((old) => ({ ...old, [name]: value }));
    };
  }

  // Batch form updates
  function setBatch(name) {
    return function ({ target: { value } }) {
      setBatchValues((old) => ({ ...old, [name]: value }));
    };
  }

  function handleChange(e) {
    setFile(e.target.files[0]);
  }

  async function getAuthenticationToken() {
    const auth = getAuth();
    const user = auth.currentUser;
    
    if (!user) {
      redirectToLogin();
      throw new Error('No user logged in');
    }

    try {
      const token = await user.getIdToken(/* forceRefresh */ true);
      setStateToken(token);
      setUserEmail(user.email);
      return token;
    } catch (error) {
      console.error('Token refresh error:', error);
      // Only redirect if it's an auth error
      if (error.code && error.code.includes('auth/')) {
        redirectToLogin();
      }
      throw error;
    }
  }

  // ----------------------------------------
  //  Single Pricing fetch
  // ----------------------------------------
  async function fetchPriceWithoutError() {
    const token = await getAuthenticationToken();
    if (!token) {
      throw new Error('No authentication token available');
    }
    const response = await getPrice(token, searchValues.cusip, searchValues.tradeType, searchValues.amount, searchValues.date, searchValues.time);
    
    // Add response validation
    if (!response) {
      throw new Error('No response from pricing service');
    }
    
    if (response.error) {
      throw new Error(response.error);
    }
    
    return response;
  }

  async function fetchPrice() {
    setIsPriceHidden(true);
    setIsPricing(true);
    setLoadingMessage('Priced at ' + (new Date()).toLocaleString());

    try {
      const response = await fetchPriceWithoutError();
      
      // Validate response structure
      if (!Array.isArray(response) || !response[0]) {
        throw new Error('Invalid response format from pricing service');
      }

      const [priceData] = response;
      const [incorporated_state_code, purpose_class, purpose_sub_class, coupon, rating, desc, predicted_yield, predicted_price, issue_date] = updateData(priceData);
      
      if (!(incorporated_state_code in statesDict)) {
        incorporated_state_code = undefined;
      }
      
      setRelatedDict({
        'state': incorporated_state_code,
        'purposeClass': purpose_class,
        'purposeSubClass': purpose_sub_class,
        'coupon': coupon,
        'rating': rating,
        'desc': defaultDesc
      });
      
      setReferenceFeatures({
        'state': incorporated_state_code,
        'purposeClass': purpose_class,
        'purposeSubClass': purpose_sub_class,
        'coupon': coupon,
        'rating': rating,
        'desc': desc,
        'datedDate': issue_date
      });
      
      setIsPricing(false);
      setIsPriceHidden(false);
      
      return [incorporated_state_code, purpose_class, purpose_sub_class, coupon, rating, desc, predicted_yield, predicted_price];
    } catch (error) {
      console.error('Pricing error:', error);
      
      // Only show logout message if it's actually an auth error
      if (error.message === 'No authentication token available' || 
          error.code?.includes('auth/') ||
          error.message.includes('unauthorized')) {
        alert(loggedOutMessage);
        redirectToLogin();
      } else {
        alert('Error getting price: ' + error.message);
      }
      
      setSimilarBondsSearchHasRun(false);
      setBlank();
      throw error;
    }
  }

  function setBlank() {
    setPricingBlank();
    setSimilarBondsRes([]);
    setIsRelatedLoading(false);
    setIsPricing(false);
    setIsBatchProcessing(false);
    setTradeHistory([]);
    setRelatedSearchVal({
      desc: '',
      coupon: '',
      radio: 'previous_day',
      issuerChoice: 'any_issuer'
    });
    setReferenceFeatures({});
  }

  function setPricingBlank() {
    setYtw('');
    setUsedDollarPriceModel(false);
    setReasonForUsingDollarPriceModel('');
    setCalcMethod('');
    setPredictedPrice('');
    setOpen(false);
    setIsPriceHidden(true);
  }

  function setRelatedDict(newVals) {
    setRelatedSearchVal((old) => ({ ...old, ...newVals }));
  }

  function updateData(content) {
    try {
      setOpen(true);
      setPredictedPrice(content.price);
      setYtw(content.ficc_ytw);

      if (content.model_used === 'dollar_price') {
        setUsedDollarPriceModel(true);
        setReasonForUsingDollarPriceModel(content.reason_for_using_dollar_price_model);
      } else {
        setUsedDollarPriceModel(false);
        setReasonForUsingDollarPriceModel('');
      }

      setCalcMethod(content.calc_date);
      setTradeHistory(content.previous_trades_features);
      setSearchCusipMaturityDate(content.maturity_date);
      setSearchCusipNextCallDate(content.next_call_date);
      setDisplayTextForYtw(content.display_text_for_ytw);
      setDisplayPriceForYtw(content.display_price);

      return [
        content.incorporated_state_code,
        content.purpose_class,
        content.purpose_sub_class,
        content.coupon,
        content.rating,
        content.security_description,
        content.ficc_ytw,
        content.price,
        content.issue_date
      ];
    } catch (error) {
      setLoadingMessage(loggedOutMessage);
      setPricingBlank();
      return [];
    }
  }

  const [searchCusipMaturityDate, setSearchCusipMaturityDate] = useState('');
  const [searchCusipNextCallDate, setSearchCusipNextCallDate] = useState('');

  // Single pricing form submission
  async function onSubmit(e) {
    e.preventDefault();
    setPricingBlank();
    await getAuthenticationToken();

    searchValues.cusip = searchValues.cusip.toUpperCase();

    const refreshRelatedTrades = isFirstTime || searchValues.cusip !== cusipForDisplay;
    if (refreshRelatedTrades) setBlank();
    setCusipForDisplay(searchValues.cusip);

    try {
      const [incStateCode, purposeClass, purposeSubClass, coupon, rating, desc, predicted_yield, predicted_price] =
        await fetchPrice();

      if (refreshRelatedTrades) {
        fetchRelated(
          searchValues.cusip,
          predicted_yield,
          predicted_price,
          relatedSearchVal.minCoupon,
          relatedSearchVal.maxCoupon,
          incStateCode,
          purposeClass,
          defaultDesc,
          rating,
          relatedSearchVal.minMaturityDate,
          relatedSearchVal.maxMaturityDate,
          relatedSearchVal.amount,
          relatedSearchVal.radio,
          relatedSearchVal.issuerChoice,
          false
        );
      }
    } catch (error) {
      setSimilarBondsSearchHasRun(false);
      setBlank();
    }
  }

  // ----------------------------------------
  //  Similar Bonds
  // ----------------------------------------
  async function fetchRelatedWithoutError(cusip, predicted_yield, predicted_price, minCoupon, maxCoupon, state, purposeClass, desc, rating, minMaturityDate, maxMaturityDate, amount, realtime, issuerChoice, userTriggered) {
    const token = await getAuthenticationToken();
    if (!token) {
      throw new Error('No authentication token available');
    }
    
    setSimilarBondsRes([]);
    
    const response = await getSimilarBonds(
      token,
      cusip,
      predicted_yield,
      predicted_price,
      minCoupon,
      maxCoupon,
      state,
      purposeClass,
      desc,
      rating,
      minMaturityDate,
      maxMaturityDate,
      amount,
      realtime,
      issuerChoice,
      userTriggered
    );

    if (!response) {
      throw new Error('No response from similar bonds service');
    }

    if (response.error) {
      throw new Error(response.error);
    }

    return response;
  }

  async function fetchRelated(
    cusip,
    predicted_yield,
    predicted_price,
    minCoupon,
    maxCoupon,
    state,
    purposeClass,
    desc,
    rating,
    minMaturityDate,
    maxMaturityDate,
    amount,
    realtime,
    issuerChoice,
    userTriggered
  ) {
    setIsRelatedLoading(true);
    try {
      const resp = await fetchRelatedWithoutError(
        cusip,
        predicted_yield,
        predicted_price,
        minCoupon,
        maxCoupon,
        state,
        purposeClass,
        desc,
        rating,
        minMaturityDate,
        maxMaturityDate,
        amount,
        realtime,
        issuerChoice,
        userTriggered
      );
      if (!resp) throw new Error(loggedOutMessage);
      if (resp.error) throw new Error(resp.error);

      setSimilarBondsSearchHasRun(true);
      setSimilarBondsRes(resp);
      setIsFirstTime(false);
    } catch (err) {
      alert(err.message || 'Error finding trades for similar bonds');
    } finally {
      setIsRelatedLoading(false);
    }
  }

  // ----------------------------------------
  //  ExpandableTradeTable transforms
  // ----------------------------------------
  function transformTradeHistory(data) {
    if (!data || !data.length) return [];
    const groupedData = {};
    data.forEach((trade) => {
      if (!trade.trade_datetime) return;
      const datePart = trade.trade_datetime.split(' ')[0];
      if (!datePart) return;
      if (!groupedData[datePart]) {
        groupedData[datePart] = {
          id: `k${datePart.replace(/[-/]/g, '')}`,
          date: datePart,
          total: 0,
          count: 0,
          dpVol: 0,
          dsVol: 0,
          ddVol: 0,
          high_price: null,
          low_price: null,
          high_yield: null,
          low_yield: null,
          details: []
        };
      }
      groupedData[datePart].total += trade.size || 0;
      groupedData[datePart].count += 1;
      if (trade.trade_type === 'P') groupedData[datePart].dpVol += trade.size || 0;
      else if (trade.trade_type === 'S') groupedData[datePart].dsVol += trade.size || 0;
      else if (trade.trade_type === 'D') groupedData[datePart].ddVol += trade.size || 0;

      // track hi/lo
      if (trade.dollar_price != null) {
        if (groupedData[datePart].high_price == null || trade.dollar_price > groupedData[datePart].high_price) {
          groupedData[datePart].high_price = trade.dollar_price;
        }
        if (groupedData[datePart].low_price == null || trade.dollar_price < groupedData[datePart].low_price) {
          groupedData[datePart].low_price = trade.dollar_price;
        }
      }
      if (trade.yield_to_worst != null) {
        if (groupedData[datePart].high_yield == null || trade.yield_to_worst > groupedData[datePart].high_yield) {
          groupedData[datePart].high_yield = trade.yield_to_worst;
        }
        if (groupedData[datePart].low_yield == null || trade.yield_to_worst < groupedData[datePart].low_yield) {
          groupedData[datePart].low_yield = trade.yield_to_worst;
        }
      }
      groupedData[datePart].details.push(trade);
    });
    return Object.values(groupedData);
  }

  function transformSimilarBondsData(data) {
    if (!data || !Array.isArray(data) || !data.length) return [];
    const groupedData = {};
    data.forEach((bond) => {
      if (!bond.cusip) return;
      if (!groupedData[bond.cusip]) {
        groupedData[bond.cusip] = {
          id: `k${bond.cusip}`,
          cusip: bond.cusip,
          state: bond.incorporated_state_code || 'N/A',
          rating: bond.rating || 'N/A',
          security_description: bond.security_description || 'N/A',
          coupon: bond.coupon || 'N/A',
          maturity_date: bond.maturity_date || 'N/A',
          avg_yield: 0,
          avg_price: 0,
          trade_count: 0,
          details: []
        };
      }
      groupedData[bond.cusip].details.push(bond);

      // update average yield
      if (bond.yield != null) {
        const count = groupedData[bond.cusip].trade_count;
        const prevSum = groupedData[bond.cusip].avg_yield * count;
        groupedData[bond.cusip].avg_yield = (prevSum + parseFloat(bond.yield)) / (count + 1);
      }
      // update average price
      if (bond.dollar_price != null) {
        const count = groupedData[bond.cusip].trade_count;
        const prevSum = groupedData[bond.cusip].avg_price * count;
        groupedData[bond.cusip].avg_price = (prevSum + parseFloat(bond.dollar_price)) / (count + 1);
      }
      groupedData[bond.cusip].trade_count += 1;
    });
    return Object.values(groupedData);
  }

  // ----------------------------------------
  //  Batch Pricing actions
  // ----------------------------------------
  function handleDisplay(e) {
    e.preventDefault();
    setShowTable(false);
    onFileUpload(false);
  }

  function handleDownload(e) {
    e.preventDefault();
    onFileUpload(true);
  }

  async function onFileUpload(isDownload) {
    if (!file) {
      alert('No file was uploaded');
      return;
    }
    isDownload ? setIsDownloadProcessing(true) : setIsBatchProcessing(true);
    await getAuthenticationToken();

    const formData = new FormData();
    formData.append('file', file);
    formData.append('access_token', stateToken);
    formData.append('amount', batchValues.quantity);
    formData.append('tradeType', batchValues.tradeType);
    if (isDownload) {
      formData.append('download', true);
      formData.append('useCachedPricedFile', true);
    }

    try {
      const response = await uploadFile(formData);
      if (isDownload) {
        const href = URL.createObjectURL(response.data);
        const link = document.createElement('a');
        link.href = href;
        link.setAttribute('download', 'preds.csv');
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(href);
      } else {
        const text = await response.data.text();
        const parsed = JSON.parse(text);
        const jsonData = JSON.parse(parsed);
        const dataArray = Object.keys(jsonData.cusip).map((key) => ({
          cusip: jsonData.cusip[key],
          quantity: jsonData.quantity[key],
          trade_type: jsonData.trade_type[key],
          ytw: jsonData.ytw[key],
          price: jsonData.price[key],
          yield_to_worst_date: jsonData.yield_to_worst_date[key],
          coupon: jsonData.coupon[key],
          security_description: jsonData.security_description[key],
          maturity_date: jsonData.maturity_date[key],
          error_message: jsonData.error_message[key]
        }));
        setTableData(dataArray);
        setShowTable(true);
      }
    } catch (error) {
      alert('Batch Pricing Error: ' + error.message);
    } finally {
      isDownload ? setIsDownloadProcessing(false) : setIsBatchProcessing(false);
    }
  }

  // ----------------------------------------
  //  Render
  // ----------------------------------------
  return (
    <Container fluid className="flex justify-content-center" style={{ fontSize: FONT_SIZE }}>
      <div>
        <Tooltip />
        {/* NavBar at top */}
        <NavBarTop message={loadingMessage} userEmail={userEmail} />
        <div
            style={{
              display: 'flex',
              justifyContent: 'flex-end',
              paddingRight: '28px',
              marginTop: '-10px',  // pull it closer to NavBarTop if needed
              marginBottom: '-25px'
            }}

          >
          <a
            href="https://analytics.ficc.ai"
            target="_blank"
            rel="noopener noreferrer"
            style={{
              backgroundColor: '#3182ce',
              color: '#fff',
              borderRadius: '6px',
              padding: '10px 16px',
              boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
              textDecoration: 'none',
              //fontWeight: 500,
              position: 'relative',
              display: 'inline-block'
            }}
            onMouseOver={(e) => e.target.style.backgroundColor = '#2c5282'}
            onMouseOut={(e) => e.target.style.backgroundColor = '#3182ce'}
          >
            Go to Analytics
          </a>
        </div>
        <Tabs
          id="controlled-tabs"
          activeKey={key}
          onSelect={(k) => setKey(k)}
          className="mb-3"
        >
          {/* Single-CUSIP Pricing Tab */}
          <Tab eventKey="pricing" title="Individual Pricing">
            <Card className="ficc-card">
              <Card.Body className="ficc-card-body">
                <TabsSearchForm
                  searchValues={searchValues}
                  set={set}
                  tradeType={tradeType}
                  onSubmit={onSubmit}
                  isPricing={isPricing}
                />

                <PricingResults
                  cusipForDisplay={cusipForDisplay}
                  isPriceHidden={isPriceHidden}
                  predictedPrice={predictedPrice}
                  displayTextForYtw={displayTextForYtw}
                  usedDollarPriceModel={usedDollarPriceModel}
                  reasonForUsingDollarPriceModel={reasonForUsingDollarPriceModel}
                  dollarPriceModelDisplayText={dollarPriceModelDisplayText}
                  ytw={ytw}
                  calcMethod={calcMethod}
                  displayPriceForYtw={displayPriceForYtw}
                  referenceFeatures={referenceFeatures}
                  searchCusipMaturityDate={searchCusipMaturityDate}
                />

                {/* Trade History */}
                {!isPriceHidden && tradeHistory?.length > 0 && (
                  <ExpandableTradeTable
                    title={`Recent trade history for ${cusipForDisplay}`}
                    data={transformTradeHistory(tradeHistory)}
                    type="tradeHistory"
                    cusip={cusipForDisplay}
                  />
                )}

                {/* Similar Bonds */}
                {!isPriceHidden && (
                  <>
                    {/* Always render the SimilarBonds form once a CUSIP is priced */}
                    <SimilarBonds
                      getAuthenticationToken={getAuthenticationToken}
                      relatedSearchVal={relatedSearchVal}
                      setRelatedSearchVal={setRelatedSearchVal}
                      similarBondsRes={similarBondsRes}
                      setSimilarBondsRes={setSimilarBondsRes}
                      fetchRelated={fetchRelated}
                      isPricing={isPricing}
                      setIsRelatedLoading={setIsRelatedLoading}
                      isRelatedLoading={isRelatedLoading}
                      noRelatedBondsFound={noRelatedBondsFound}
                      setNoRelatedBondsFound={setNoRelatedBondsFound}
                      similarBondsSearchHasRun={similarBondsSearchHasRun}
                      setSimilarBondsSearchHasRun={setSimilarBondsSearchHasRun}
                      searchValCusip={searchValues.cusip}
                      predictedYield={ytw}
                      predictedPrice={predictedPrice}
                    />

                    {/* If still loading, show a spinner right here in Pricing */}
                    {isRelatedLoading && (
                      <div className="d-flex justify-content-center mt-3">
                        <Spinner animation="border" role="status">
                          <span className="visually-hidden">Loading Similar Bonds...</span>
                        </Spinner>
                      </div>
                    )}

                    {/* Otherwise, if we have run a search and have results, show them */}
                    {!isRelatedLoading && similarBondsSearchHasRun && similarBondsRes?.length > 0 && (
                      <ExpandableTradeTable
                        title="Recent trades for similar bonds"
                        data={transformSimilarBondsData(similarBondsRes)}
                        type="similarBonds"
                        cusip={cusipForDisplay}
                      />
                    )}
                  </>
                )}
              </Card.Body>
            </Card>
          </Tab>

          {/* Batch Pricing Tab */}
          <Tab eventKey="batch" title="Batch Pricing">
            <BatchPricing
              handleDisplay={handleDisplay}
              handleChange={handleChange}
              handleDownload={handleDownload}
              file={file}
              batchValues={batchValues}
              setBatch={setBatch}
              isBatchProcessing={isBatchProcessing}
              isDownloadProcessing={isDownloadProcessing}
              showTable={showTable}
              tableData={tableData}
              tableProps={{
                getTableProps,
                getTableBodyProps,
                headerGroups,
                page,
                prepareRow,
                canPreviousPage,
                canNextPage,
                pageOptions,
                pageCount,
                gotoPage,
                nextPage,
                previousPage,
                setPageSize,
                state: { pageIndex, pageSize }
              }}
              isPricing={isPricing}
              tradeType={tradeType}
            />
          </Tab>
        </Tabs>
      </div>
    </Container>
  );
}

export default Pricing;
