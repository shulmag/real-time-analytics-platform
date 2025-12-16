/**
 * API service for ficc analytics.
 * Handles API requests with Firebase authentication.
 */

import { API_URL, DEFAULT_MATURITIES, EMAIL_ONLY_AUTH_ENABLED } from '../config';
import { auth } from './auth';

class ApiService {
  constructor() {
    // Simple token cache
    this.tokenCache = {
      token: null,
      timestamp: null,
      // Firebase tokens last 1 hour, but we'll refresh after 50 minutes
      expiryTime: 50 * 60 * 1000 // 50 minutes in milliseconds
    };
  }

  /**
   * Get Firebase authentication token with caching to avoid quota issues
   * @param {boolean} forceRefresh
   * @returns {Promise<string|null>}
   */
  async getAuthToken(forceRefresh = false) {
    try {
      const now = Date.now();
      if (
        !forceRefresh &&
        this.tokenCache.token &&
        this.tokenCache.timestamp &&
        (now - this.tokenCache.timestamp) < this.tokenCache.expiryTime
      ) {
        console.log('Using cached Firebase token');
        return this.tokenCache.token;
      }

      const currentUser = auth.getCurrentUser();

      if (currentUser) {
        console.log('Found current user:', currentUser.email);
        try {
          const token = await currentUser.getIdToken(false);
          console.log('Token retrieved successfully');
          console.log("Firebase ID Token (first 30 chars):", token.substring(0, 30) + "...");

          if (EMAIL_ONLY_AUTH_ENABLED) {
            const providerId = currentUser.providerData[0]?.providerId;
            console.log(`Auth provider: ${providerId || 'email-link'}`);
          }

          this.tokenCache.token = token;
          this.tokenCache.timestamp = now;
          return token;
        } catch (error) {
          console.error('Error getting token, trying fallback:', error);
          if (this.tokenCache.token && error.code === 'auth/quota-exceeded') {
            console.log('Using cached token due to quota error');
            return this.tokenCache.token;
          }
          throw error;
        }
      } else {
        const userData = localStorage.getItem('user');
        if (userData) {
          console.log('No Firebase user but found user in localStorage');
        } else {
          console.log('No user found in Firebase or localStorage');
        }
        return null;
      }
    } catch (error) {
      console.error('Error in getAuthToken:', error);
      return null;
    }
  }

  /**
   * Fetch Treasury yield curve data from backend API
   */
  async getYieldCurves(
    type = 'realtime',
    startDate = null,
    endDate = null,
    maturities = DEFAULT_MATURITIES,
    isRefresh = false
  ) {
    const params = new URLSearchParams();
    params.append('type', type);
    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);
    if (maturities && maturities.length > 0) params.append('maturities', maturities.join(','));
    if (isRefresh) params.append('refresh', 'true');

    const headers = {};
    try {
      const token = await this.getAuthToken();
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
        console.log('Added cached Firebase auth token to request');
      }
    } catch (error) {
      console.error('Error getting Firebase token, continuing without authentication:', error);
    }

    console.log(`Making API call to ${API_URL}/api/yield-curves with Authorization header`);
    console.log("Making API call", {url: `${API_URL}/api/yield-curves?${params.toString()}`,headers});
    const response = await fetch(`${API_URL}/api/yield-curves?${params.toString()}`, {
      headers,
      mode: 'cors'
    });

    if (!response.ok) {
      throw new Error(`Error fetching yield curves: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Fetch real-time yield curve Plot data from Pricing server
   */
  async getRealtimeYieldCurvePlotFromPricing(date, time) {

    const PRICING_API_URL = 'https://api.ficc.ai';

    let token = '';
    try {
      token = await this.getAuthToken();
    } catch (err) {
      console.warn('No Firebase token available for pricing request:', err);
    }

    const url = `${PRICING_API_URL}/api/yield?access_token=${token}&date=${date}&time=${time}`;

    const response = await fetch(url, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      mode: 'cors'
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch pricing curve for Plot: ${response.statusText}`);
    }
    return await response.json();
  }

  /**
   * Fetch real-time yield curve table data from Pricing server
   */
  async getRealtimeYieldCurveTableFromPricing(date, time) {
    const PRICING_API_URL = 'https://api.ficc.ai';

    let token = '';
    try {
      token = await this.getAuthToken();
    } catch (err) {
      console.warn('No Firebase token available for pricing request:', err);
    }

    const url = `${PRICING_API_URL}/api/realtimeyieldcurve?access_token=${token}&date=${date}&time=${time}`;

    const response = await fetch(url, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      mode: 'cors'
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch pricing curve for Table: ${response.statusText}`);
    }
    return await response.json();
  }

  /**
   * Fetch market strength metrics based on MSRB trade data
   */
  async getMarketMetrics(date = null, isRefresh = false) {
    const params = new URLSearchParams();
    if (date) params.append('date', date);
    if (isRefresh) params.append('refresh', 'true');

    const authToken = await this.getAuthToken();

    const headers = { 'Content-Type': 'application/json' };
    if (authToken) headers['Authorization'] = `Bearer ${authToken}`;

    const response = await fetch(`${API_URL}/api/market-metrics?${params.toString()}`, { headers });
    if (!response.ok) {
      throw new Error(`Error fetching market metrics: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Fetch investment grade bond spread data
   */
  async getSpreads({ maturities = [], quantities = [], days = 15, isRefresh = false } = {}) {
    const params = new URLSearchParams();
    if (isRefresh) params.append('refresh', 'true');
    if (maturities.length) params.append('maturities', maturities.join(','));
    if (quantities.length) params.append('quantities', quantities.join(','));
    if (days) params.append('days', String(days));

    const headers = {};
    try {
      const token = await this.getAuthToken();
      if (token) headers['Authorization'] = `Bearer ${token}`;
    } catch (e) {
      console.error('Error getting Firebase token for spreads:', e);
    }

    const r = await fetch(`${API_URL}/api/spreads?${params.toString()}`, {
      headers,
      mode: 'cors'
    });
    if (!r.ok) throw new Error(`Error fetching spreads: ${r.statusText}`);
    return r.json();
  }

  /**
   * Fetch recap cards (today) for Muni Market Stats
   */
  async getMuniMarketStats() {
    const authToken = await this.getAuthToken();
    const headers = { 'Content-Type': 'application/json' };
    if (authToken) headers['Authorization'] = `Bearer ${authToken}`;

    const response = await fetch(`${API_URL}/api/muni-market-stats`, { headers });
    if (!response.ok) {
      const text = await response.text().catch(() => '');
      throw new Error(`muni-market-stats failed: ${response.status} ${text}`);
    }
    return await response.json();
  }

  /**
   * Fetch last 10 business days D/S/P counts for stacked bar chart
   */
  async getMuniMarketStats10d() {
    const authToken = await this.getAuthToken();
    const headers = { 'Content-Type': 'application/json' };
    if (authToken) headers['Authorization'] = `Bearer ${authToken}`;

    const resp = await fetch(`${API_URL}/api/muni-market-stats-10d`, { headers });
    if (!resp.ok) {
      const text = await resp.text().catch(() => '');
      throw new Error(`muni-market-stats-10d failed: ${resp.status} ${text}`);
    }
    return await resp.json();
  }

  /**
   * NEW: Fetch Top 10 issues for current & previous business day
   * @param {Object} opts
   * @param {'seasoned'|'new'} [opts.issue_type] Optional filter; omit for all types
   * @returns {Promise<{
   *   current_as_of_date: string|null,
   *   previous_as_of_date: string|null,
   *   issue_type: string|null,
   *   current: Array, previous: Array
   * }>}
   */
  async getMuniTopIssues({ issue_type } = {}) {
    const authToken = await this.getAuthToken();
    const headers = { 'Content-Type': 'application/json' };
    if (authToken) headers['Authorization'] = `Bearer ${authToken}`;

    const params = new URLSearchParams();
    if (issue_type) params.set('issue_type', issue_type);
    const qs = params.toString();
    const url = `${API_URL}/api/muni-top-issues${qs ? `?${qs}` : ''}`;

    const resp = await fetch(url, { headers, mode: 'cors' });
    if (!resp.ok) {
      const text = await resp.text().catch(() => '');
      throw new Error(`muni-top-issues failed: ${resp.status} ${text}`);
    }
    return await resp.json();
  }

  /**
   * Clear the token cache when logging out
   */
  clearTokenCache() {
    console.log('Clearing token cache');
    this.tokenCache = {
      token: null,
      timestamp: null,
      expiryTime: this.tokenCache.expiryTime
    };
  }


  /**
   * Fetch AAA Benchmark (5y & 10y) intraday series (fixed 5-minute smoothing).
   * Returns { dates: [...], data: { "YYYY-MM-DD": [ { time, "5", "10" }, ... ] } }
   */
async getAAABenchmark() {
  // Reuse the same auth pattern as other endpoints
  const headers = {};
  try {
    const token = await this.getAuthToken();
      if (token) headers['Authorization'] = `Bearer ${token}`;
  } catch (e) {
      console.warn('Proceeding without auth for AAA benchmark:', e);
  }

  const resp = await fetch(`${API_URL}/api/aaa-benchmark`, {
    headers,
    mode: 'cors'
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => '');
         throw new Error(`aaa-benchmark failed: ${resp.status} ${text}`);
    }
  return await resp.json();
  }
}

// Create and export a singleton instance
const apiService = new ApiService();
export { apiService };
export default apiService;
