// src/components/LoadingFallback.jsx
import React from 'react';

const LoadingFallback = () => {
  return (
    <div style={{ 
      display: 'flex', 
      justifyContent: 'center', 
      alignItems: 'center', 
      height: '100vh',
      flexDirection: 'column',
      gap: '20px',
      backgroundColor: '#fafafa'
    }}>
      <div className="loading-spinner"></div>
      <p style={{ 
        color: '#666', 
        fontSize: '18px',
        fontWeight: '500'
      }}>
        Loading Insulyn AI...
      </p>
      
      <style>
        {`
          .loading-spinner {
            width: 60px;
            height: 60px;
            border: 4px solid #e3f2fd;
            border-top: 4px solid #1976d2;
            border-radius: 50%;
            animation: spin 1s linear infinite;
          }
          
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}
      </style>
    </div>
  );
};

export default LoadingFallback;
