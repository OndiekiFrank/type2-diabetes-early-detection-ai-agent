import React, { useState, lazy, Suspense } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

// Lazy load components for better performance
const Header = lazy(() => import('./components/Header'));
const Footer = lazy(() => import('./components/Footer'));
const Home = lazy(() => import('./pages/Home'));
const DiabetesTest = lazy(() => import('./pages/DiabetesTest'));
const ChatBot = lazy(() => import('./pages/ChatBot'));
const VoiceChat = lazy(() => import('./pages/VoiceChat'));
const DietPlan = lazy(() => import('./pages/DietPlan'));
const EmergencyCheck = lazy(() => import('./pages/EmergencyCheck'));

// Loading component
const LoadingFallback = () => (
  <div className="loading-container">
    <div className="loading-spinner"></div>
    <p className="loading-text">Loading Insulyn AI...</p>
    <style>{`
      .loading-container {
        display: flex; 
        justify-content: center; 
        align-items: center; 
        height: 100vh;
        flex-direction: column;
        gap: 16px;
        background-color: #fafafa;
      }
      .loading-spinner {
        width: 60px; 
        height: 60px; 
        border: 4px solid #e3f2fd;
        border-top: 4px solid #1976d2;
        border-radius: 50%;
        animation: spin 1s linear infinite;
      }
      .loading-text {
        color: #666; 
        font-size: 18px;
        font-weight: 500;
      }
      @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
    `}</style>
  </div>
);

// Error Boundary Component
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('App Error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-container">
          <h2>Oops! Something went wrong</h2>
          <p>We're sorry for the inconvenience. Please try refreshing the page.</p>
          <button 
            onClick={() => window.location.reload()}
            className="reload-button"
          >
            Reload Application
          </button>
          
          {process.env.NODE_ENV === 'development' && this.state.error && (
            <details className="error-details">
              <summary>Error Details (Development)</summary>
              <pre>{this.state.error.toString()}</pre>
            </details>
          )}
          <style>{`
            .error-container {
              padding: 40px 20px; 
              text-align: center;
              max-width: 500px;
              margin: 50px auto;
              border: 1px solid #ffcdd2;
              border-radius: 8px;
              background-color: #ffebee;
            }
            .error-container h2 { 
              color: #d32f2f; 
              margin-bottom: 20px; 
            }
            .error-container p { 
              margin-bottom: 20px; 
              color: #666; 
            }
            .reload-button {
              padding: 10px 20px;
              background-color: #d32f2f;
              color: white;
              border: none;
              border-radius: 4px;
              cursor: pointer;
              font-size: 16px;
              transition: background-color 0.3s;
            }
            .reload-button:hover {
              background-color: #b71c1c;
            }
            .error-details { 
              margin-top: 30px; 
              text-align: left; 
            }
            .error-details summary { 
              cursor: pointer; 
              margin-bottom: 10px; 
            }
            .error-details pre { 
              background: #f5f5f5; 
              padding: 15px; 
              border-radius: 4px;
              overflow: auto;
              font-size: 12px;
              text-align: left;
            }
          `}</style>
        </div>
      );
    }

    return this.props.children;
  }
}

// 404 Component
const NotFoundPage = () => (
  <div className="not-found-container">
    <h1>404 - Page Not Found</h1>
    <p>The page you're looking for doesn't exist or has been moved.</p>
    <a href="/" className="home-link">Return to Home</a>
    <style>{`
      .not-found-container {
        text-align: center; 
        padding: 50px 20px;
        max-width: 600px;
        margin: 0 auto;
        min-height: 60vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
      }
      .not-found-container h1 { 
        color: #1976d2; 
        margin-bottom: 20px; 
      }
      .not-found-container p { 
        font-size: 18px; 
        margin-bottom: 30px; 
        color: #666; 
      }
      .home-link {
        display: inline-block;
        padding: 12px 30px;
        background-color: #1976d2;
        color: white;
        text-decoration: none;
        border-radius: 6px;
        font-size: 16px;
        font-weight: 500;
        transition: background-color 0.3s;
      }
      .home-link:hover {
        background-color: #1565c0;
      }
    `}</style>
  </div>
);

// Theme configuration
const theme = createTheme({
  palette: {
    primary: {
      main: '#2E7D32',
      light: '#4CAF50',
      dark: '#1B5E20',
    },
    secondary: {
      main: '#FF6B35',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: { fontWeight: 600 },
    h5: { fontWeight: 500 },
    h6: { fontWeight: 500 },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        },
      },
    },
  },
});

function App() {
  const [language, setLanguage] = useState('english');

  return (
    <ErrorBoundary>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Router>
          <div className="App">
            <Suspense fallback={<div className="header-placeholder"></div>}>
              <Header language={language} setLanguage={setLanguage} />
            </Suspense>
            
            <main className="main-content">
              <Suspense fallback={<LoadingFallback />}>
                <Routes>
                  <Route path="/" element={<Home language={language} />} />
                  <Route path="/test" element={<DiabetesTest language={language} />} />
                  <Route path="/chat" element={<ChatBot language={language} />} />
                  <Route path="/voice-chat" element={<VoiceChat language={language} />} />
                  <Route path="/diet-plan" element={<DietPlan language={language} />} />
                  <Route path="/emergency" element={<EmergencyCheck language={language} />} />
                  <Route path="*" element={<NotFoundPage />} />
                </Routes>
              </Suspense>
            </main>
            
            <Suspense fallback={<div className="footer-placeholder"></div>}>
              <Footer language={language} />
            </Suspense>
            
            <style>{`
              .header-placeholder {
                height: 64px; 
                background: #2E7D32;
              }
              .footer-placeholder {
                height: 60px; 
                background: #f5f5f5;
              }
              .main-content {
                min-height: 80vh; 
                padding-top: 20px;
              }
            `}</style>
          </div>
        </Router>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default React.memo(App);