# FICC AI Chrome Extension for Schwab

This Chrome extension adds real-time bond prices from api.ficc.ai to CUSIPs displayed on Schwab bond search pages.

## Features

- Automatically detects CUSIPs on Schwab bond search pages
- Fetches prices from api.ficc.ai using batch pricing API
- Displays prices inline with visual indicators
- Always fetches fresh prices from API (no caching)
- Supports dynamic content updates (single-page application)

## Installation

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" in the top right
3. Click "Load unpacked"
4. Select the `ficc-chrome-extension` directory
5. The extension icon will appear in your Chrome toolbar

## Setup

1. Click the FICC extension icon in your toolbar
2. Enter your FICC AI credentials:
   - Email: Your FICC AI account email
   - Password: Your API password
3. Click "Save Credentials"
4. Navigate to a Schwab bond search page
5. Prices will automatically be added to all visible CUSIPs

## How It Works

1. **Content Script**: Runs on Schwab pages and extracts CUSIPs
2. **Background Script**: Handles API calls to api.ficc.ai
3. **Popup**: Allows users to configure credentials
4. **Fresh Data**: Always fetches latest prices from API

## API Integration

The extension uses the FICC AI batch pricing endpoint:
- Endpoint: `https://api.ficc.ai/api/batchpricing`
- Method: POST
- Parameters:
  - `username`: User email
  - `password`: API password
  - `cusipList`: Comma-separated CUSIPs
  - `quantityList`: Face values (defaults to 100)
  - `tradeTypeList`: Trade types (defaults to 'S' - Sale to Customer)

## Trade Types

- `P`: Purchase from Customer
- `S`: Sale to Customer (default)
- `D`: Inter-Dealer

## Visual Indicators

- **Blue Badge**: FICC price successfully fetched
- **Highlighted Row**: CUSIP has FICC pricing
- **Status Bar**: Shows loading and completion status

## Troubleshooting

1. **No prices showing**: 
   - Check your credentials in the popup
   - Ensure you're on a Schwab bond search page
   - Check the browser console for errors

2. **Prices not updating**:
   - Refresh the page to fetch fresh prices

3. **Extension not working**:
   - Make sure the extension is enabled in Chrome
   - Check that you have the correct permissions

## Development

To modify the extension:

1. Edit the source files
2. Go to `chrome://extensions/`
3. Click the refresh icon on the FICC AI extension card
4. Reload the Schwab page to see changes

## Files

- `manifest.json`: Extension configuration
- `content.js`: Extracts CUSIPs and injects prices
- `background.js`: Handles API calls and caching
- `popup.html/js`: User interface for credentials
- `styles.css`: Styling for injected prices

## Future Enhancements (Step 2)

- Firebase authentication integration
- User preferences for trade types and quantities
- Advanced pricing options
- Historical price charts
- Export functionality

## Icons

To add custom icons, create PNG files:
- `icon16.png`: 16x16 pixels
- `icon48.png`: 48x48 pixels  
- `icon128.png`: 128x128 pixels

## Security

- Credentials are stored locally in Chrome's secure storage
- API calls are made over HTTPS
- No data is sent to third parties
- No data is cached locally

## Support

For issues or questions, contact gil@ficc.ai