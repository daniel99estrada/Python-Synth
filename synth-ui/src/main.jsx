import React from 'react'
import ReactDOM from 'react-dom/client'
import SynthController from './components/SynthController'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <div className="min-h-screen bg-gray-100 py-8">
      <SynthController />
    </div>
  </React.StrictMode>,
)