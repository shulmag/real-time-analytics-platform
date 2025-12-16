// Step 2: Extract CUSIPs and display FICC prices
console.log('💰 FICC Price Display loaded');

// DEBUG MODE - Set to true to disable auto-retry
const DEBUG_MODE = true;
const MAX_RUNS = DEBUG_MODE ? 1 : 999; // Run only once in debug mode
let runCount = 0;

// Global variables for tracking
let mutationTimeout;
let isProcessing = false;
const processedCusips = new Set();

// Extract CUSIPs from the page
function extractCUSIPs() {
  const cusips = [];
  const cusipMap = new Map();
  
  // Find all rows with CUSIPs
  const rows = document.querySelectorAll('tr');
  rows.forEach(row => {
    const text = row.textContent;
    const cusipMatch = text.match(/CUSIP\s+([A-Z0-9]{9})/i);
    
    if (cusipMatch) {
      const cusip = cusipMatch[1].toUpperCase();
      if (!cusipMap.has(cusip)) {
        cusips.push(cusip);
        cusipMap.set(cusip, row);
      }
    }
  });
  
  console.log(`Found ${cusips.length} unique CUSIPs`);
  return { cusips, cusipMap };
}

// Call FICC API through background script
async function fetchPrices(cusips) {
  console.log(`Fetching prices for ${cusips.length} CUSIPs...`);
  
  return new Promise((resolve) => {
    try {
      chrome.runtime.sendMessage(
        {
          action: 'fetchPrices',
          cusips: cusips
        },
        (response) => {
          console.log('API Response:', response); // Debug logging
          
          if (chrome.runtime.lastError) {
            console.error('Chrome runtime error:', chrome.runtime.lastError);
            resolve(null);
            return;
          }
          
          if (response && response.success && response.prices) {
            // Convert the prices object to the expected format
            const cusipArray = [];
            const priceArray = [];
            const ytwArray = [];
            const errorArray = [];
            
            cusips.forEach(cusip => {
              cusipArray.push(cusip);
              const priceData = response.prices[cusip];
              console.log(`Price data for ${cusip}:`, priceData); // Debug logging
              
              if (priceData) {
                priceArray.push(priceData.price || null);
                ytwArray.push(priceData.yield || null);
                errorArray.push(priceData.error || null);
              } else {
                priceArray.push(null);
                ytwArray.push(null);
                errorArray.push('No price data');
              }
            });
            
            const result = {
              cusip: cusipArray,
              price: priceArray,
              ytw: ytwArray,
              error_message: errorArray
            };
            
            console.log('Converted result:', result); // Debug logging
            resolve(result);
          } else {
            console.error('Failed to fetch prices:', response?.error || 'Unknown error');
            resolve(null);
          }
        }
      );
    } catch (error) {
      console.error('Error sending message to background script:', error);
      resolve(null);
    }
  });
}

// Display prices in the page
function displayPrices(cusipMap, apiResponse) {
  if (!apiResponse || !apiResponse.cusip) {
    console.error('Invalid API response');
    return;
  }
  
  let successCount = 0;
  const cusipCount = apiResponse.cusip.length;
  
  for (let i = 0; i < cusipCount; i++) {
    const cusip = apiResponse.cusip[i];
    const price = apiResponse.price ? apiResponse.price[i] : null;
    const ytw = apiResponse.ytw ? apiResponse.ytw[i] : null;
    const error = apiResponse.error_message ? apiResponse.error_message[i] : null;
    
    console.log(`Processing CUSIP ${cusip}: price=${price}, ytw=${ytw}, error=${error}`);
    
    const row = cusipMap.get(cusip);
    if (!row) continue;
    
    // Mark as processed
    processedCusips.add(cusip);
    
    // Check if price already exists
    if (row.querySelector('.ficc-price-badge')) {
      continue; // Skip if already has price
    }
    
    // Find the CUSIP cell
    const cells = row.querySelectorAll('td');
    let cusipCell = null;
    
    for (const cell of cells) {
      if (cell.textContent.includes(cusip)) {
        cusipCell = cell;
        break;
      }
    }
    
    if (!cusipCell) continue;
    
    // Create price badge
    const badge = document.createElement('span');
    badge.className = 'ficc-price-badge';
    
    if (price && !error) {
      badge.textContent = `FICC: $${price.toFixed(3)} | YTW: ${ytw.toFixed(2)}%`;
      badge.classList.add('success');
      
      // Highlight row
      row.classList.add('ficc-price-row');
      successCount++;
    } else {
      badge.textContent = 'FICC: No price';
      badge.classList.add('error');
    }
    
    // Insert badge after CUSIP
    cusipCell.appendChild(badge);
  }
  
  console.log(`Successfully displayed ${successCount} prices`);
  
  // Update summary
  if (successCount > 0) {
    showSummary(`✅ FICC prices loaded: ${successCount} bonds priced`);
  }
}

// Show summary message
function showSummary(message) {
  const existing = document.getElementById('ficc-summary');
  if (existing) existing.remove();
  
  const summary = document.createElement('div');
  summary.id = 'ficc-summary';
  summary.className = 'ficc-summary';
  summary.textContent = `✅ ${message}`;
  document.body.appendChild(summary);
  
  // Auto-hide after 5 seconds
  setTimeout(() => {
    summary.style.opacity = '0';
    setTimeout(() => summary.remove(), 500);
  }, 5000);
}

// Main function
async function main() {
  // Check run count
  runCount++;
  console.log(`🏃 Main function run #${runCount}${DEBUG_MODE ? ' (DEBUG MODE)' : ''}`);
  
  if (runCount > MAX_RUNS) {
    console.log('⛔ Reached maximum run limit, stopping');
    return;
  }
  
  return new Promise((resolve) => {
    // Get credentials from storage to check if they're set
    chrome.storage.local.get(['ficc_username', 'ficc_password'], async (result) => {
      if (!result.ficc_username || !result.ficc_password) {
        showSummary('⚠️ Please set credentials in extension popup');
        resolve();
        return;
      }
      
      // Extract CUSIPs
      const { cusips, cusipMap } = extractCUSIPs();
      
      // Filter out already processed CUSIPs
      const newCusips = cusips.filter(cusip => !processedCusips.has(cusip));
      
      if (newCusips.length === 0) {
        console.log('No new CUSIPs to process');
        resolve();
        return;
      }
      
      console.log(`Processing ${newCusips.length} new CUSIPs`);
      
      // Show loading indicator
      showSummary(`Loading prices for ${newCusips.length} CUSIPs...`);
      
      const batchSize = 1000;
      for (let i = 0; i < newCusips.length; i += batchSize) {
        const batch = newCusips.slice(i, i + batchSize);
        const batchMap = new Map();
        batch.forEach(cusip => batchMap.set(cusip, cusipMap.get(cusip)));
        
        const response = await fetchPrices(batch);
        if (response) {
          displayPrices(batchMap, response);
        }
        
        // Small delay between batches
        if (i + batchSize < newCusips.length) {
          await new Promise(r => setTimeout(r, 1000));
        }
      }
      
      resolve();
    });
  });
}

// Test background script connection
chrome.runtime.sendMessage({action: 'test'}, (response) => {
  if (chrome.runtime.lastError) {
    console.error('Background script not responding:', chrome.runtime.lastError.message);
  } else {
    console.log('Background script is working:', response);
  }
});

// Run when page loads
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    setTimeout(main, 1000);
  });
} else {
  setTimeout(main, 1000);
}

// Re-run when page content changes with debouncing
const observer = new MutationObserver(() => {
  // Check if debug mode is on
  if (DEBUG_MODE) {
    console.log('🛑 MutationObserver triggered but DEBUG_MODE is on - not re-running');
    return;
  }
  
  // Don't trigger if we're already processing
  if (isProcessing) return;
  
  clearTimeout(mutationTimeout);
  mutationTimeout = setTimeout(() => {
    // Check if there are new rows without prices
    const rows = document.querySelectorAll('tr');
    let hasNewCusips = false;
    
    rows.forEach(row => {
      const cusipMatch = row.textContent.match(/CUSIP\s+([A-Z0-9]{9})/i);
      if (cusipMatch) {
        const cusip = cusipMatch[1].toUpperCase();
        // Only process if we haven't seen this CUSIP before and it doesn't have a badge
        if (!processedCusips.has(cusip) && !row.querySelector('.ficc-price-badge')) {
          hasNewCusips = true;
        }
      }
    });
    
    if (hasNewCusips) {
      console.log('New CUSIPs detected, fetching prices...');
      isProcessing = true;
      main().finally(() => {
        isProcessing = false;
      });
    }
  }, 5000); // Wait 5 seconds before checking for changes
});

observer.observe(document.body, {
  childList: true,
  subtree: true
});

// Add manual refresh button for debugging
if (DEBUG_MODE) {
  const debugButton = document.createElement('button');
  debugButton.textContent = '🔄 Refresh FICC Prices';
  debugButton.style.cssText = `
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 10000;
    padding: 10px 20px;
    background: #007bff;
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-weight: bold;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
  `;
  debugButton.onclick = () => {
    console.log('Manual refresh triggered');
    processedCusips.clear(); // Clear processed set to force re-fetch
    runCount = 0; // Reset run count
    main();
  };
  document.body.appendChild(debugButton);
}