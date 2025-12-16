// Popup script for FICC AI Chrome Extension

document.addEventListener('DOMContentLoaded', async () => {
  const form = document.getElementById('credentialsForm');
  const usernameInput = document.getElementById('username');
  const passwordInput = document.getElementById('password');
  const tradeTypeInput = document.getElementById('tradeType');
  const quantityInput = document.getElementById('quantity');
  const statusDiv = document.getElementById('status');
  const loadingDiv = document.getElementById('loading');
  const statsDiv = document.getElementById('stats');
  const cachedCountSpan = document.getElementById('cachedCount');
  const apiCallsCountSpan = document.getElementById('apiCallsCount');
  const refreshButton = document.getElementById('refreshButton');
  
  // Load existing settings
  const settings = await getStoredSettings();
  if (settings.username) {
    usernameInput.value = settings.username;
    // Don't show the actual password, just indicate it's set
    if (settings.password) {
      passwordInput.placeholder = 'Password is set (enter to update)';
      passwordInput.value = ''; // Clear any existing value
    }
    showStatus('Settings loaded', 'info');
    await updateStats();
  }
  
  // Load trade type and quantity settings
  if (settings.tradeType) {
    tradeTypeInput.value = settings.tradeType;
  }
  if (settings.quantity) {
    quantityInput.value = settings.quantity;
  }
  
  // Handle form submission
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const username = usernameInput.value.trim();
    const password = passwordInput.value.trim();
    const tradeType = tradeTypeInput.value;
    const quantity = quantityInput.value;
    
    if (!username) {
      showStatus('Please enter your email', 'error');
      return;
    }
    
    // Only require password if it's not already stored
    if (!password && !settings.password) {
      showStatus('Please enter your password', 'error');
      return;
    }
    
    if (!tradeType) {
      showStatus('Please select a trade type', 'error');
      return;
    }
    
    if (!quantity || quantity < 1) {
      showStatus('Please enter a valid quantity', 'error');
      return;
    }
    
    // Show loading
    loadingDiv.classList.add('active');
    statusDiv.style.display = 'none';
    
    try {
      // Store all settings (use existing password if no new one provided)
      const passwordToUse = password || settings.password;
      await storeSettings(username, passwordToUse, tradeType, quantity);
      
      // Test the credentials with a simple API call
      const testResult = await testCredentials(username, passwordToUse, tradeType, quantity);
      
      loadingDiv.classList.remove('active');
      
      if (testResult.success) {
        showStatus('Settings saved successfully!', 'success');
        // Clear password field but keep placeholder
        passwordInput.placeholder = 'Password is set (enter to update)';
        passwordInput.value = '';
        await updateStats();
      } else {
        showStatus('Invalid credentials. Please check and try again.', 'error');
      }
    } catch (error) {
      loadingDiv.classList.remove('active');
      showStatus('Error saving settings: ' + error.message, 'error');
    }
  });
  
  // Get stored settings
  async function getStoredSettings() {
    return new Promise((resolve) => {
      chrome.runtime.sendMessage({ action: 'getSettings' }, (response) => {
        resolve(response || { username: '', password: '', tradeType: 'S', quantity: '100' });
      });
    });
  }
  
  // Store settings
  async function storeSettings(username, password, tradeType, quantity) {
    return new Promise((resolve, reject) => {
      chrome.runtime.sendMessage(
        { 
          action: 'setSettings',
          username: username,
          password: password,
          tradeType: tradeType,
          quantity: quantity
        },
        (response) => {
          if (response && response.success) {
            resolve();
          } else {
            reject(new Error('Failed to store settings'));
          }
        }
      );
    });
  }
  
  // Test credentials with a simple API call through background script
  async function testCredentials(username, password, tradeType, quantity) {
    try {
      // First store the settings
      await storeSettings(username, password, tradeType, quantity);
      
      // Test with a single CUSIP through the background script
      const testCusip = '64971XQM3';
      
      return new Promise((resolve) => {
        chrome.runtime.sendMessage(
          {
            action: 'fetchPrices',
            cusips: [testCusip],
            tradeType: tradeType,
            quantity: quantity
          },
          (response) => {
            if (response && response.success) {
              resolve({ success: true, data: response });
            } else {
              resolve({ success: false, error: response?.error || 'Invalid credentials' });
            }
          }
        );
      });
    } catch (error) {
      console.error('Error testing credentials:', error);
      return { success: false, error: error.message };
    }
  }
  
  // Show status message
  function showStatus(message, type) {
    statusDiv.textContent = message;
    statusDiv.className = 'status ' + type;
    statusDiv.style.display = 'block';
    
    // Auto-hide success messages after 3 seconds
    if (type === 'success') {
      setTimeout(() => {
        statusDiv.style.display = 'none';
      }, 3000);
    }
  }
  
  // Update statistics
  async function updateStats() {
    chrome.storage.local.get(null, (items) => {
      let apiCallsToday = 0;
      
      Object.keys(items).forEach(key => {
        if (key === 'api_calls_today') {
          apiCallsToday = items[key] || 0;
        }
      });
      
      cachedCountSpan.textContent = '0 (caching disabled)';
      apiCallsCountSpan.textContent = apiCallsToday;
      
      if (apiCallsToday > 0) {
        statsDiv.classList.add('visible');
      }
    });
  }
  
  // Update stats on load
  updateStats();
  
  // Handle refresh button click
  refreshButton.addEventListener('click', async () => {
    try {
      // Show loading state
      loadingDiv.classList.add('active');
      statusDiv.style.display = 'none';
      refreshButton.disabled = true;
      refreshButton.textContent = '🔄 Refreshing...';
      
      // Send message to content script to refresh prices
      const response = await new Promise((resolve) => {
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
          if (tabs[0]) {
            chrome.tabs.sendMessage(tabs[0].id, { action: 'refreshPrices' }, (response) => {
              resolve(response);
            });
          } else {
            resolve({ success: false, error: 'No active tab found' });
          }
        });
      });
      
      loadingDiv.classList.remove('active');
      refreshButton.disabled = false;
      refreshButton.textContent = '🔄 Refresh FICC Prices';
      
      if (response && response.success) {
        showStatus('FICC prices refreshed successfully!', 'success');
        await updateStats();
      } else {
        showStatus('Error refreshing prices: ' + (response?.error || 'Unknown error'), 'error');
      }
    } catch (error) {
      loadingDiv.classList.remove('active');
      refreshButton.disabled = false;
      refreshButton.textContent = '🔄 Refresh FICC Prices';
      showStatus('Error refreshing prices: ' + error.message, 'error');
    }
  });
  
  // Refresh stats every 5 seconds while popup is open
  setInterval(updateStats, 5000);
});