/**
 * Description: API service for communicating with the Muni Market Monitor backend.
 * Handles authentication and data fetching for municipal bond price data.
 */

import axios from 'axios';
import { auth } from './auth';

const api = axios.create({
  baseURL: process.env.NODE_ENV === 'production' 
    ? 'https://monitor-964018767272.us-central1.run.app' 
    : 'http://localhost:8080'
});

export const getPrices = async () => {
  const user = auth.currentUser;
  if (!user) {
    throw new Error('Not authenticated');
  }

  const formData = new FormData();
  formData.append('username', user.email);
  formData.append('password', await user.getIdToken());
  formData.append('access_token', await user.getIdToken());

  // Changed to GET since that's what the server expects
  const response = await api.get('/prices', { params: {
    username: user.email,
    access_token: await user.getIdToken()
  }});
  return response.data;
};
