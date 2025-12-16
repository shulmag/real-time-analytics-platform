// Background service worker for the FICC AI Chrome Extension
// Handles API calls to api.ficc.ai

// API Configuration
const API_BASE_URL = 'https://api.ficc.ai/api';
const BATCH_PRICING_ENDPOINT = '/batchpricing';

// Storage keys
const STORAGE_KEYS = {
  USERNAME: 'ficc_username',
  PASSWORD: 'ficc_password',
  TRADE_TYPE: 'ficc_trade_type',
  QUANTITY: 'ficc_quantity'
};

// Get stored settings
async function getSettings() {
  return new Promise((resolve) => {
    chrome.storage.local.get([
      STORAGE_KEYS.USERNAME, 
      STORAGE_KEYS.PASSWORD, 
      STORAGE_KEYS.TRADE_TYPE, 
      STORAGE_KEYS.QUANTITY
    ], (result) => {
      resolve({
        username: result[STORAGE_KEYS.USERNAME] || '',
        password: result[STORAGE_KEYS.PASSWORD] || '',
        tradeType: result[STORAGE_KEYS.TRADE_TYPE] || 'S',
        quantity: result[STORAGE_KEYS.QUANTITY] || '25'
      });
    });
  });
}

// Store settings
async function storeSettings(username, password, tradeType, quantity) {
  return new Promise((resolve) => {
    chrome.storage.local.set({
      [STORAGE_KEYS.USERNAME]: username,
      [STORAGE_KEYS.PASSWORD]: password,
      [STORAGE_KEYS.TRADE_TYPE]: tradeType,
      [STORAGE_KEYS.QUANTITY]: quantity
    }, resolve);
  });
}

// Fetch prices from FICC API
async function fetchPricesFromAPI(cusips, tradeType = null, quantity = null) {
  const settings = await getSettings();
  
  console.log('=== FICC API Call Debug ===');
  console.log('Settings:', { 
    username: settings.username, 
    hasPassword: !!settings.password,
    tradeType: tradeType || settings.tradeType,
    quantity: quantity || settings.quantity
  });
  console.log('CUSIPs to fetch:', cusips);
  
  if (!settings.username || !settings.password) {
    throw new Error('Please configure your FICC AI credentials in the extension popup');
  }
  
  // Build form data with multiple parameters for each CUSIP
  const formData = new URLSearchParams();
  formData.append('username', settings.username);
  formData.append('password', settings.password);
  
  // Use provided parameters or fall back to stored settings
  const finalTradeType = tradeType || settings.tradeType;
  const finalQuantity = quantity || settings.quantity;
  
  // Add each CUSIP as a separate parameter
  cusips.forEach(cusip => {
    formData.append('cusipList', cusip);
    formData.append('quantityList', finalQuantity);
    formData.append('tradeTypeList', finalTradeType);
  });
  
  console.log('Request URL:', API_BASE_URL + BATCH_PRICING_ENDPOINT);
  console.log('Request body:', formData.toString());
  
  try {
    // Make the API request
    const response = await fetch(API_BASE_URL + BATCH_PRICING_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: formData
    });
    
    console.log('Response status:', response.status);
    console.log('Response headers:', Object.fromEntries(response.headers.entries()));
    
    // Get response as text first to see raw response
    const responseText = await response.text();
    console.log('Raw response:', responseText);
    
    if (!response.ok) {
      throw new Error(`API request failed with status ${response.status}: ${responseText}`);
    }
    
    // Try to parse as JSON (double parse like in pricing.jsx)
    let data;
    try {
      const parsed = JSON.parse(responseText);
      data = JSON.parse(parsed);  // Double parse like in your React app
      console.log('Parsed response:', data);
    } catch (e) {
      console.error('Failed to parse response as JSON:', e);
      throw new Error(`Invalid JSON response: ${responseText}`);
    }
    
    return data;
  } catch (error) {
    console.error('API request error:', error);
    throw error;
  }
}

// Process price response from API
function processPriceResponse(apiResponse, cusips) {
  console.log('=== Processing API Response ===');
  console.log('Response type:', typeof apiResponse);
  console.log('Response keys:', Object.keys(apiResponse));
  
  const prices = {};
  
  // Use the same parsing logic as pricing.jsx lines 703-714
  if (apiResponse && apiResponse.cusip && typeof apiResponse.cusip === 'object') {
    console.log('Found cusip object with keys:', Object.keys(apiResponse.cusip).slice(0, 10));
    
    // Map each CUSIP index to create price objects (like in pricing.jsx)
    Object.keys(apiResponse.cusip).forEach((key) => {
      const cusip = apiResponse.cusip[key];
      const price = apiResponse.price ? apiResponse.price[key] : null;
      const ytw = apiResponse.ytw ? apiResponse.ytw[key] : null;
      const errorMessage = apiResponse.error_message ? apiResponse.error_message[key] : null;
      
      if (cusip) {
        prices[cusip] = {
          price: price,
          yield: ytw,
          error: errorMessage,
          timestamp: new Date().toISOString()
        };
      }
    });
  } else {
    console.error('Response does not have expected cusip structure');
    console.log('Available response keys:', Object.keys(apiResponse));
  }
  
  console.log(`Successfully processed ${Object.keys(prices).length} prices out of ${cusips.length} requested CUSIPs`);
  return prices;
}

// Handle messages from content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('Background script received message:', request);
  
  if (request.action === 'test') {
    sendResponse({ success: true, message: 'Background script is working' });
    return;
  } else if (request.action === 'fetchPrices') {
    handleFetchPrices(request.cusips, request.tradeType, request.quantity).then(sendResponse);
    return true; // Will respond asynchronously
  } else if (request.action === 'setSettings') {
    storeSettings(request.username, request.password, request.tradeType, request.quantity).then(() => {
      sendResponse({ success: true });
    });
    return true;
  } else if (request.action === 'getSettings') {
    getSettings().then(sendResponse);
    return true;
  }
});

// Handle fetch prices request
async function handleFetchPrices(cusips, tradeType = null, quantity = null) {
  try {
    console.log(`=== Fetching prices for ${cusips.length} CUSIPs ===`);
    console.log('CUSIPs:', cusips);
    console.log('Trade Type:', tradeType);
    console.log('Quantity:', quantity);
    
    // Always fetch from API, no caching
    const apiResponse = await fetchPricesFromAPI(cusips, tradeType, quantity);
    const apiPrices = processPriceResponse(apiResponse, cusips);
    
    console.log('Final prices object:', apiPrices);
    console.log('Number of prices returned:', Object.keys(apiPrices).length);
    
    return {
      success: true,
      prices: apiPrices,
      cached: 0,
      fetched: Object.keys(apiPrices).length
    };
  } catch (error) {
    console.error('Error in handleFetchPrices:', error);
    return {
      success: false,
      error: error.message
    };
  }
}

// Track API calls for debugging
let apiCallCount = 0;
chrome.runtime.onMessage.addListener((request) => {
  if (request.action === 'fetchPrices') {
    apiCallCount++;
    console.log(`API call #${apiCallCount} at ${new Date().toISOString()}`);
  }
});

console.log('FICC AI Price Extension background script loaded (no-cache version)');