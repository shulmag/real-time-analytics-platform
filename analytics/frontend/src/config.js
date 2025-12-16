// API configuration
// export const API_URL = 'http://localhost:8000';
export const API_URL = 'https://us-central1-eng-reactor-287421.cloudfunctions.net/analytics-server-v2';

// Authentication configuration
export const EMAIL_ONLY_AUTH_ENABLED = true; // Enable email-only auth with magic links
export const AUTH_PERSISTENCE_DAYS = 30; // Number of days to persist authentication
export const AUTH_REDIRECT_URL = window.location.origin; // For email link redirect
export const APP_DISPLAY_NAME = 'ficc.ai Analytics'; // Used in email templates

// Default maturities to display
export const DEFAULT_MATURITIES = [5, 10, 15, 20, 25, 26, 27, 28, 29, 30];

// Chart colors
export const CHART_COLORS = {
  yesterday: '#555555', // Medium gray instead of dark gray
  today: '#3182ce',     // Blue
  buys: '#38a169',      // Green
  sells: '#e53e3e'      // Red
};

// Date format options
export const DATE_FORMAT_OPTIONS = {
  short: {
    month: '2-digit',
    day: '2-digit',
    year: 'numeric'
  },
  long: {
    month: 'long',
    day: 'numeric',
    year: 'numeric'
  },
  time: {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false
  }
};

// Market hours
export const MARKET_HOURS = {
  open: 9,  // 9:00 AM
  close: 16, // 4:00 PM
  dataAvailableHour: 9,
  dataAvailableMinute: 35 // Data available after 9:35 AM ET
};