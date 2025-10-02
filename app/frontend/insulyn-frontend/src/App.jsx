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
  <div style={{ 
    display: 'flex', 
    justifyContent: 'center', 
    alignItems: 'center', 
    height: '100vh',
    flexDirection: 'column',
    gap: '16px',
    backgroundColor: '#fafafa'
  }}>
    <div style={{ 
      width: '60px', 
      height: '60px', 
      border: '4px solid #e3f2fd',
      borderTop: '4px solid #1976d2',
      borderRadius: '50%',
      animation: 'spin 1s linear infinite'
    }}></div>
    <p style={{ 
      color: '#666', 
      fontSize: '18px',
      fontWeight: '500'
    }}>Loading Insulyn AI...</p>
    <style>
      {`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}
    </style>
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
          
          {process.env.NODE_ENV === 'development' && this.state.error && (
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
                {this.state.error.toString()}
              </pre>
            </details>
          )}
        </div>
      );
    }

    return this.props.children;
  }
}

// 404 Component
const NotFoundPage = () => (
  <div style={{ 
    textAlign: 'center', 
    padding: '50px 20px',
    maxWidth: '600px',
    margin: '0 auto',
    minHeight: '60vh',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center'
  }}>
    <h1 style={{ color: '#1976d2', marginBottom: '20px' }}>404 - Page Not Found</h1>
    <p style={{ fontSize: '18px', marginBottom: '30px', color: '#666' }}>
      The page you're looking for doesn't exist or has been moved.
    </p>
    <a 
      href="/" 
      style={{
        display: 'inline-block',
        padding: '12px 30px',
        backgroundColor: '#1976d2',
        color: 'white',
        textDecoration: 'none',
        borderRadius: '6px',
        fontSize: '16px',
        fontWeight: '500',
        transition: 'background-color 0.3s'
      }}
      onMouseOver={(e) => e.target.style.backgroundColor = '#1565c0'}
      onMouseOut={(e) => e.target.style.backgroundColor = '#1976d2'}
    >
      Return to Home
    </a>
  </div>
);

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
    h4: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 500,
    },
    h6: {
      fontWeight: 500,
    },
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

  // Memoize the context value to prevent unnecessary re-renders
  const appContextValue = React.useMemo(() => ({
    language,
    setLanguage,
  }), [language]);

  return (
    <ErrorBoundary>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Router>
          <div className="App">
            <Suspense fallback={<div style={{ height: '64px', background: '#2E7D32' }}></div>}>
              <Header language={language} setLanguage={setLanguage} />
            </Suspense>
            <main style={{ minHeight: '80vh', paddingTop: '20px' }}>
              <Suspense fallback={<LoadingFallback />}>
                <Routes>
                  <Route path="/" element={<Home language={language} />} />
                  <Route path="/test" element={<DiabetesTest language={language} />} />
                  <Route path="/chat" element={<ChatBot language={language} />} />
                  <Route path="/voice-chat" element={<VoiceChat language={language} />} />
                  <Route path="/diet-plan" element={<DietPlan language={language} />} />
                  <Route path="/emergency" element={<EmergencyCheck language={language} />} />
                  {/* 404 Catch-all route */}
                  <Route path="*" element={<NotFoundPage />} />
                </Routes>
              </Suspense>
            </main>
            <Suspense fallback={<div style={{ height: '60px', background: '#f5f5f5' }}></div>}>
              <Footer language={language} />
            </Suspense>
          </div>
        </Router>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default React.memo(App);