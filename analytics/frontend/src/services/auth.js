/**
 * Authentication service for ficc analytics.
 * Provides email/password and magic link (passwordless) authentication via Firebase.
 */

import { 
    signInWithEmailAndPassword, 
    signOut, 
    onAuthStateChanged,
    sendSignInLinkToEmail,
    isSignInWithEmailLink,
    signInWithEmailLink,
    setPersistence,
    browserLocalPersistence
  } from 'firebase/auth';
  
  import { auth as firebaseAuth } from './firebase';
  import { EMAIL_ONLY_AUTH_ENABLED, AUTH_REDIRECT_URL, APP_DISPLAY_NAME } from '../config';
  
  // Email storage key for email link sign-in
  const EMAIL_FOR_LINK_KEY = 'emailForSignIn';
  
  // Configure Firebase auth persistence (30 days)
  setPersistence(firebaseAuth, browserLocalPersistence)
    .catch(error => {
      console.error('Error setting auth persistence:', error);
    });
  
  // Action code settings for email link authentication
  // Simplified to work without Dynamic Links
  const actionCodeSettings = {
    // URL you want to redirect back to after sign-in
    url: AUTH_REDIRECT_URL,
    // This must be true for email link sign-in
    handleCodeInApp: true
  };
  
  // Wrap Firebase auth in our own auth service interface
  export const auth = {
    // Traditional email/password sign in
    signIn: async (email, password) => {
      try {
        const userCredential = await signInWithEmailAndPassword(firebaseAuth, email, password);
        return {
          success: true,
          email: userCredential.user.email
        };
      } catch (error) {
        throw error;
      }
    },
    
    // Send magic link for passwordless sign in
    sendSignInLink: async (email) => {
      try {
        console.log("Sending sign-in link to email:", email);
        console.log("Using action code settings:", actionCodeSettings);
        
        // Try to send email link
        await sendSignInLinkToEmail(firebaseAuth, email, actionCodeSettings);
        console.log("Sign-in link sent successfully");
        
        // Save the email locally so you don't need to ask the user again
        // if they open the link on the same device
        localStorage.setItem(EMAIL_FOR_LINK_KEY, email);
        console.log(`Email saved to localStorage with key: ${EMAIL_FOR_LINK_KEY}`);
        
        return {
          success: true,
          message: 'Sign-in link sent to email'
        };
      } catch (error) {
        console.error("Error sending sign-in link:", error);
        throw error;
      }
    },
    
    // Check if the URL is a sign-in link
    isSignInLink: (url) => {
      console.log("Checking if URL is a sign-in link:", url);
      try {
        const result = isSignInWithEmailLink(firebaseAuth, url);
        console.log("Firebase isSignInWithEmailLink result:", result);
        
        // Also check for our fallback link format
        // This is a simple check for URL parameters that might indicate a verification link
        const isFallbackLink = url.includes('mode=verifyEmail') || 
                               url.includes('mode=signIn') || 
                               url.includes('oobCode=');
        
        console.log("Is fallback verification link:", isFallbackLink);
        
        return result || isFallbackLink;
      } catch (error) {
        console.error("Error checking if URL is sign-in link:", error);
        return false;
      }
    },
    
    // Complete sign in with email link
    signInWithLink: async (email, link) => {
      try {
        console.log("Completing sign-in with email link");
        console.log("Email:", email);
        console.log("Link:", link);
        
        // Check if it's a standard email link
        const isStandardEmailLink = isSignInWithEmailLink(firebaseAuth, link);
        
        if (!isStandardEmailLink) {
          console.log("Not a standard email link");
          throw new Error("This does not appear to be a valid sign-in link. Please request a new one.");
        }
        
        // Attempt to sign in with the email link
        try {
          console.log("Attempting Firebase email link sign-in");
          
          // Special handling for auth/email-already-in-use errors
          // Check if the user exists first by trying password-less methods
          try {
            const userCredential = await signInWithEmailLink(firebaseAuth, email, link);
            
            console.log("Sign-in successful. User credential:", userCredential);
            
            // Clear email from storage after successful sign-in
            localStorage.removeItem(EMAIL_FOR_LINK_KEY);
            console.log(`Removed email from localStorage with key: ${EMAIL_FOR_LINK_KEY}`);
            
            return {
              success: true,
              email: userCredential.user.email,
              isNewUser: userCredential.additionalUserInfo?.isNewUser || false
            };
          } catch (signInError) {
            console.error("Initial sign-in attempt error:", signInError);
            
            // If this is an email-already-in-use error, the user probably exists with password
            if (signInError.code === 'auth/email-already-in-use') {
              // Here we could try to get a credential and sign in with it
              // But for simplicity, we'll just guide the user to password login
              throw new Error("This email is already registered. Please sign in with your password.");
            } else if (signInError.code === 'auth/invalid-action-code') {
              throw new Error("The link is invalid or has expired. Please request a new link.");
            } else {
              throw signInError;
            }
          }
        } catch (error) {
          console.error("Error with email link sign-in:", error);
          throw error;
        }
      } catch (error) {
        console.error("Error in signInWithLink:", error);
        throw error;
      }
    },
    
    // Get email from storage for email link sign-in
    getEmailFromStorage: () => {
      const email = localStorage.getItem(EMAIL_FOR_LINK_KEY);
      console.log(`Retrieved email from localStorage with key ${EMAIL_FOR_LINK_KEY}:`, email);
      return email;
    },
    
    // Sign out
    signOut: () => {
      localStorage.removeItem(EMAIL_FOR_LINK_KEY);
      return signOut(firebaseAuth);
    },
    
    // Auth state listener
    onAuthStateChanged: (callback) => {
      return onAuthStateChanged(firebaseAuth, callback);
    },
    
    // Get current user
    getCurrentUser: () => {
      return firebaseAuth.currentUser;
    },
    
    // Feature flag for email-only auth
    isEmailOnlyAuthEnabled: () => {
      return EMAIL_ONLY_AUTH_ENABLED;
    }
  };
  
  export default auth;