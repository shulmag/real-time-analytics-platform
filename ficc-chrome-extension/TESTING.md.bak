# Testing Instructions for FICC Chrome Extension

## Setup

1. **Generate Icons**
   - Open `create_icons.html` in a browser
   - Right-click each canvas and save as:
     - icon16.png
     - icon48.png  
     - icon128.png
   - Save all in the `ficc-chrome-extension` directory

2. **Load Extension in Chrome**
   - Open Chrome and go to `chrome://extensions/`
   - Enable "Developer mode" (toggle in top right)
   - Click "Load unpacked"
   - Select the `ficc-chrome-extension` folder
   - The extension should appear in your extensions list

3. **Configure Credentials**
   - Click the FICC extension icon in Chrome toolbar
   - Enter your FICC API credentials:
     - Email: Your email (e.g., gil@ficc.ai)
     - Password: Your API password
   - Click "Save Credentials"
   - You should see "Credentials saved successfully!"

## Testing

1. **Navigate to Schwab Bond Page**
   - Go to: https://client.schwab.com/Areas/Trade/FixedIncomeSearch/FISearch.aspx/Municipals
   - Or navigate through Schwab: Trade → Bonds → Municipals

2. **Verify Extension is Working**
   - Check browser console (F12) for messages like:
     - "FICC AI Price Extension loaded on Schwab page"
     - "Found X unique CUSIPs on the page"
   - Look for blue FICC price badges appearing next to CUSIPs

3. **What to Expect**
   - Extension automatically extracts CUSIPs from the page
   - Fetches prices in batches of 50
   - Displays prices with blue badges
   - Shows comparison with Schwab prices where available
   - Prices are cached for 5 minutes

## Troubleshooting

### No Prices Showing
1. Check credentials are saved (click extension icon)
2. Verify you're on a Schwab bond page with CUSIPs
3. Open browser console and check for errors
4. Try refreshing the page

### API Errors
1. Verify credentials are correct
2. Check network tab in DevTools for API calls
3. Look for 401 (unauthorized) or 500 (server) errors

### Extension Not Loading
1. Make sure all files are in place
2. Check for errors in chrome://extensions/
3. Try reloading the extension
4. Check that icons are present

## Debug Mode

To see detailed logs:
1. Open Chrome DevTools (F12)
2. Go to Console tab
3. Filter by "FICC" to see extension messages

## API Response Format

The extension expects responses in this format:
```json
[
  {
    "price": 100.123,
    "yield": 3.456,
    "spread": 150
  }
]
```

## Known Issues

1. **Dynamic Content**: The extension monitors for page changes but may miss some updates
2. **Rate Limiting**: Large numbers of CUSIPs may hit API rate limits
3. **Cache**: Prices are cached for 5 minutes - refresh page to force new fetch

## Testing Checklist

- [ ] Icons generated and saved
- [ ] Extension loads without errors
- [ ] Popup opens and accepts credentials
- [ ] Credentials save successfully
- [ ] Extension activates on Schwab page
- [ ] CUSIPs are detected
- [ ] API calls are made
- [ ] Prices are displayed
- [ ] Cache works (check within 5 minutes)
- [ ] Page mutations are handled

## Sample Test CUSIPs

From the Schwab page:
- 735389XD5
- 54659LBH6
- 438670W80
- 88236QAE3
- 64988YJH1

## Next Steps (Phase 2)

- Firebase authentication integration
- User preferences for trade types
- Quantity customization
- Bulk export functionality
- Historical price tracking