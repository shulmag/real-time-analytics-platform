// Content script that runs on Schwab pages
// Extracts CUSIPs and adds pricing information
// Add this at the beginning of content.js
const style = document.createElement('style');
style.textContent = `
  .ficc-price-injected {
    background-color: #2563eb;
    color: white;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 11px;
    margin-top: 4px;
    display: inline-block;
  }
  
  .ficc-price-column {
    background-color: #f8f9fa;
    border-left: 2px solid #2563eb;
  }
  
  .ficc-price-cell {
    text-align: right;
    font-weight: normal;
    color: #2563eb;
    background-color: #f8f9fa;
    padding: 4px 8px;
    border: 1px solid #dee2e6;
  }
  

  
  .ficc-price-header {
    background-color: #2563eb;
    color: white;
    text-align: center;
    font-weight: bold;
    padding: 8px 4px;
  }
`;
document.head.appendChild(style);

console.log('FICC AI Price Extension loaded on Schwab page');

// Configuration
const BATCH_SIZE = 2000; // Process CUSIPs in batches
const RETRY_ATTEMPTS = 0;
const RETRY_DELAY = 1000; // milliseconds

// Add FICC.AI Price column to the table next to Price column
function addFICCPriceColumn() {
  const table = document.querySelector('table.table-results');
  if (!table) {
    console.log('Schwab bond search table not found');
    return false;
  }
  
  const thead = table.querySelector('thead');
  const tbody = table.querySelector('tbody');
  
  if (!thead || !tbody) {
    console.log('Table header or body not found');
    return false;
  }
  
  // Check if ficc.ai Price column already exists
  const existingFiccHeader = thead.querySelector('th.ficc-price-header');
  if (existingFiccHeader) {
    console.log('ficc.ai Price column already exists, skipping creation');
    return true;
  }
  
  // Find the Price column header
  const headerRow = thead.querySelector('tr');
  if (!headerRow) {
    console.log('Table header row not found');
    return false;
  }
  
  // Find the Price column and insert FICC.AI Price after it
  const headers = headerRow.querySelectorAll('th');
  let priceHeaderIndex = -1;
  
  headers.forEach((header, index) => {
    if (header.textContent.trim().toLowerCase() === 'price') {
      priceHeaderIndex = index;
    }
  });
  
  if (priceHeaderIndex === -1) {
    console.log('Price column not found, adding ficc.ai Price column at the end');
    // If Price column not found, add at the end
    const ficcHeader = document.createElement('th');
    ficcHeader.className = 'ficc-price-header';
    ficcHeader.textContent = 'ficc.ai Price (yield)';
    ficcHeader.style.minWidth = '100px';
    headerRow.appendChild(ficcHeader);
  } else {
    // Insert ficc.ai Price column after the Price column
    const ficcHeader = document.createElement('th');
    ficcHeader.className = 'ficc-price-header';
    ficcHeader.textContent = 'ficc.ai Price (yield)';
    ficcHeader.style.minWidth = '100px';
    
    // Insert after the Price column
    const priceHeader = headers[priceHeaderIndex];
    priceHeader.parentNode.insertBefore(ficcHeader, priceHeader.nextSibling);
  }
  
  // Add data column for each row
  const rows = tbody.querySelectorAll('tr');
  rows.forEach(row => {
    // Check if ficc.ai Price cell already exists in this row
    const existingFiccCell = row.querySelector('.ficc-price-cell');
    if (existingFiccCell) {
      console.log('ficc.ai Price cell already exists in row, skipping creation');
      return;
    }
    
    const ficcCell = document.createElement('td');
    ficcCell.className = 'ficc-price-cell';
    ficcCell.textContent = 'Loading...';
    ficcCell.setAttribute('data-ficc-status', 'loading');
    
    // Find the Price cell in this row and insert ficc.ai Price after it
    const cells = row.querySelectorAll('td');
    let priceCellIndex = -1;
    
    // Look for the Price column specifically (the one with the main price, not yield)
    cells.forEach((cell, index) => {
      const cellText = cell.textContent.trim();
      // Look for cells that contain the main price (like 100.30700)
      if (cellText.match(/^\d{2,3}\.\d{5}$/)) {
        priceCellIndex = index;
      }
    });
    
    if (priceCellIndex === -1) {
      // If Price cell not found, add at the end
      row.appendChild(ficcCell);
    } else {
      // Insert ficc.ai Price cell after the Price cell
      const priceCell = cells[priceCellIndex];
      priceCell.parentNode.insertBefore(ficcCell, priceCell.nextSibling);
    }
  });
  
  console.log(`Added ficc.ai Price column to ${rows.length} rows`);
  return true;
}

// Extract CUSIPs from the page
function extractCUSIPs() {
  const cusips = new Set();
  
  // Look for CUSIP patterns in the Schwab table structure
  // The Schwab page shows CUSIPs after "CUSIP" text in the same row
  const rows = document.querySelectorAll('tr');
  
  rows.forEach(row => {
    const text = row.textContent;
    // Look for "CUSIP" followed by a 9-character alphanumeric code
    const cusipMatches = text.matchAll(/CUSIP\s+([A-Z0-9]{9})/g);
    for (const match of cusipMatches) {
      cusips.add(match[1]);
    }
    
    // Also look for standalone 9-character codes that look like CUSIPs
    const cells = row.querySelectorAll('td');
    cells.forEach(cell => {
      const cellText = cell.textContent.trim();
      // Check if this is a CUSIP pattern (9 characters, alphanumeric)
      if (/^[A-Z0-9]{9}$/.test(cellText)) {
        // Additional validation: CUSIPs typically have specific patterns
        // First 6 chars are issuer, next 2 are issue, last is check digit
        if (/^[0-9A-Z]{6}[A-Z0-9]{2}[0-9]$/.test(cellText)) {
          cusips.add(cellText);
        }
      }
    });
  });

  console.log(`Found ${cusips.size} unique CUSIPs on the page`);
  return Array.from(cusips);
}

function findCUSIPLocations() {
  const locations = new Map();
  
  // Find the Schwab bond search table
  const table = document.querySelector('table.table-results');
  if (!table) {
    console.log('Schwab bond search table not found');
    return locations;
  }
  
  // Find all rows in the table body
  const rows = table.querySelectorAll('tbody tr');
  
  rows.forEach(row => {
    // Look for CUSIP in the row text (it's usually in the description or as a separate element)
    const rowText = row.textContent;
    const cusipMatches = rowText.match(/CUSIP\s+([A-Z0-9]{9})/g);
    
    if (cusipMatches) {
      cusipMatches.forEach(match => {
        const cusip = match.replace('CUSIP ', '');
        
        if (!locations.has(cusip)) {
          locations.set(cusip, []);
        }
        
        locations.get(cusip).push({
          row: row,
          cusipCell: null, // We don't need the specific CUSIP cell anymore
          priceCell: null,
          yieldCell: null
        });
      });
    }
  });
  
  console.log(`Found ${locations.size} CUSIP locations in the table`);
  return locations;
}

// Send message to background script to fetch prices
async function fetchPrices(cusips) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage(
      {
        action: 'fetchPrices',
        cusips: cusips
      },
      (response) => {
        resolve(response);
      }
    );
  });
}

// Inject prices into the new FICC Price column
function injectPrices(cusipLocations, prices) {
  let injectedCount = 0;
  
  console.log('=== DEBUG: injectPrices called ===');
  console.log('cusipLocations size:', cusipLocations.size);
  console.log('prices object:', prices);
  
  cusipLocations.forEach((locations, cusip) => {
    const priceData = prices[cusip];
    console.log(`Processing CUSIP ${cusip}, priceData:`, priceData);
    
    if (!priceData) {
      console.log(`No price data for CUSIP ${cusip}`);
      return;
    }
    
    locations.forEach(location => {
      const { row } = location;
      
      // Find the ficc.ai price cell in this row
      let ficcCell = row.querySelector('.ficc-price-cell');
      console.log(`Looking for ficc-price-cell in row for CUSIP ${cusip}, found:`, ficcCell);
      
      if (!ficcCell) {
        // Try alternative selector
        ficcCell = row.querySelector('td[data-ficc-status="loading"]');
        console.log(`Trying alternative selector for CUSIP ${cusip}, found:`, ficcCell);
      }
      
      if (!ficcCell) {
        console.log(`FICC price cell not found for row with CUSIP ${cusip}`);
        return;
      }
      
      // Clear existing content and reset to loading state
      ficcCell.className = 'ficc-price-cell';
      ficcCell.textContent = 'Loading...';
      ficcCell.setAttribute('data-ficc-status', 'loading');
      // Remove any existing difference indicators
      const existingDiffIndicator = ficcCell.querySelector('div');
      if (existingDiffIndicator) {
        existingDiffIndicator.remove();
      }
      
      // Format price data
      let priceText = '';
      let tooltipText = 'FICC AI Price';
      
      if (priceData.price || priceData.dollar_price) {
        const price = priceData.price || priceData.dollar_price;
        priceText = parseFloat(price).toFixed(3);
        tooltipText += ` - Price: ${price}`;
      }
      
      if (priceData.yield || priceData.ytm) {
        const yieldValue = priceData.yield || priceData.ytm;
        if (priceText) {
          priceText += ` (${parseFloat(yieldValue).toFixed(3)}%)`;
        } else {
          priceText = `${parseFloat(yieldValue).toFixed(3)}%`;
        }
        tooltipText += ` - Yield: ${yieldValue}%`;
      }
      
      if (priceData.spread || priceData.yield_spread) {
        const spread = priceData.spread || priceData.yield_spread;
        tooltipText += ` - Spread: ${spread}bps`;
      }
      
      // Update the FICC price cell
      console.log(`Setting ficcCell.textContent to: "${priceText || 'N/A'}"`);
      ficcCell.textContent = priceText || 'N/A';
      ficcCell.title = tooltipText;
      ficcCell.setAttribute('data-ficc-status', 'loaded');
      console.log(`Updated ficcCell, new textContent: "${ficcCell.textContent}"`);
      
      // Add visual indicator for comparison with Schwab price
      const schwabPriceCell = row.querySelector('td:has(span.floatRight)');
      if (schwabPriceCell && priceData.price) {
        const schwabPriceText = schwabPriceCell.textContent.match(/\d+\.\d+/);
        if (schwabPriceText) {
          const schwabPrice = parseFloat(schwabPriceText[0]);
          const ficcPrice = parseFloat(priceData.price);
          const diff = ficcPrice - schwabPrice;
          
          if (Math.abs(diff) > 0.001) { // Only show if there's a meaningful difference
            const diffIndicator = document.createElement('div');
            diffIndicator.style.fontSize = '10px';
            diffIndicator.style.color = diff > 0 ? '#28a745' : '#dc3545';
            diffIndicator.textContent = diff > 0 ? `+${diff.toFixed(3)}` : `${diff.toFixed(3)}`;
            ficcCell.appendChild(diffIndicator);
          }
        }
      }
      
      injectedCount++;
    });
  });
  
  // Show status badge
  showStatusBadge(`Loaded ${injectedCount} FICC prices`);
  
  console.log(`Injected prices for ${injectedCount} CUSIPs`);
  return injectedCount;
}

// Show a status badge temporarily
function showStatusBadge(message) {
  // Remove existing badge if any
  const existingBadge = document.querySelector('.ficc-badge');
  if (existingBadge) {
    existingBadge.remove();
  }
  
  // Create new badge
  const badge = document.createElement('div');
  badge.className = 'ficc-badge show';
  badge.textContent = `✅ ${message}`;
  document.body.appendChild(badge);
  
  // Remove after 3 seconds
  setTimeout(() => {
    badge.classList.remove('show');
    setTimeout(() => badge.remove(), 500);
  }, 3000);
}

// Process CUSIPs in batches
async function processCUSIPBatch(cusips, cusipLocations) {
  const batches = [];
  for (let i = 0; i < cusips.length; i += BATCH_SIZE) {
    batches.push(cusips.slice(i, i + BATCH_SIZE));
  }
  
  console.log(`Processing ${batches.length} batches of CUSIPs`);
  
  for (let i = 0; i < batches.length; i++) {
    const batch = batches[i];
    console.log(`Processing batch ${i + 1}/${batches.length} with ${batch.length} CUSIPs`);
    
    try {
      const response = await fetchPrices(batch);
      
      if (response && response.success && response.prices) {
        injectPrices(cusipLocations, response.prices);
      } else {
        console.error('Failed to fetch prices for batch:', response?.error || 'Unknown error');
      }
    } catch (error) {
      console.error('Error processing batch:', error);
    }
    
    // Add delay between batches to avoid overwhelming the API
    if (i < batches.length - 1) {
      await new Promise(resolve => setTimeout(resolve, 500));
    }
  }
}

// Main function to orchestrate the price injection
async function main() {
  console.log('Starting FICC AI price injection...');
  
  // Add FICC Price column to the table
  const columnAdded = addFICCPriceColumn();
  if (!columnAdded) {
    console.log('Failed to add FICC Price column');
    return;
  }
  
  // Extract CUSIPs from the page
  const cusips = extractCUSIPs();
  if (cusips.length === 0) {
    console.log('No CUSIPs found on the page');
    return;
  }
  
  // Find CUSIP locations in the DOM
  const cusipLocations = findCUSIPLocations();
  console.log(`Found ${cusipLocations.size} CUSIP locations in the DOM`);
  
  // Process CUSIPs and inject prices
  await processCUSIPBatch(cusips, cusipLocations);
  
  console.log('FICC AI price injection complete');
}

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'refreshPrices') {
    console.log('Received refresh prices request from popup');
    main().then(() => {
      sendResponse({ success: true, message: 'Prices refreshed successfully' });
    }).catch((error) => {
      console.error('Error refreshing prices:', error);
      sendResponse({ success: false, error: error.message });
    });
    return true; // Will respond asynchronously
  }
});

// Wait for the page to load before running
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', main);
} else {
  // DOM is already loaded
  main();
}

// Disabled mutation observer to prevent endless retries
// const observer = new MutationObserver((mutations) => {
//   const hasTableChanges = mutations.some(mutation => {
//     return mutation.type === 'childList' && 
//            (mutation.target.tagName === 'TABLE' || 
//             mutation.target.tagName === 'TBODY' ||
//             mutation.target.querySelector('table'));
//   });
//   
//   if (hasTableChanges) {
//     console.log('Page content changed, re-running FICC AI price injection');
//     main();
//   }
// });
// observer.observe(document.body, { childList: true, subtree: true });