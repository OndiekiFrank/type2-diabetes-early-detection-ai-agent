// src/components/ErrorBoundary.jsx
import React from 'react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({
      error: error,
      errorInfo: errorInfo
    });
    
    // Log to error reporting service
    console.error('Error caught by boundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ 
          padding: '40px 20px', 
          textAlign: 'center',
          maxWidth: '500px',
          margin: '50px auto',
          border: '1px solid #ffcdd2',
          borderRadius: '8px',
          backgroundColor: '#ffebee'
        }}>
          <h2 style={{ color: '#d32f2f', marginBottom: '20px' }}>
            Oops! Something went wrong
          </h2>
          <p style={{ marginBottom: '20px', color: '#666' }}>
            We're sorry for the inconvenience. Please try refreshing the page.
          </p>
          <button 
            onClick={() => window.location.reload()}
            style={{
              padding: '10px 20px',
              backgroundColor: '#d32f2f',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '16px'
            }}
          >
            Reload Application
          </button>
          
          {process.env.NODE_ENV === 'development' && (
            <details style={{ marginTop: '30px', textAlign: 'left' }}>
              <summary style={{ cursor: 'pointer', marginBottom: '10px' }}>
                Error Details (Development)
              </summary>
              <pre style={{ 
                background: '#f5f5f5', 
                padding: '15px', 
                borderRadius: '4px',
                overflow: 'auto',
                fontSize: '12px',
                textAlign: 'left'
              }}>
                {this.state.error && this.state.error.toString()}
                <br />
                {this.state.errorInfo.componentStack}
              </pre>
            </details>
          )}
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
