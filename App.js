// src/App.js

import React, { useState } from 'react';
import './App.css';
// Remove the import below
// import backgroundImage from './images/background.png';

function App() {
  const [time, setTime] = useState('');
  const [amount, setAmount] = useState('');
  const [result, setResult] = useState('');

  const handleSubmit = () => {
    // Simulate fraud detection logic here
    if (time && amount) {
      // Example logic for demonstration
      if (amount > 1000) {
        setResult('Fraudulent Transaction');
      } else {
        setResult('Non-Fraudulent Transaction');
      }
    } else {
      setResult('Please enter time and amount.');
    }
  };

  return (
    <div className="app">
      <div className="background-overlay">
        <h1>Credit Card Fraud Detection</h1>
        <div className="input-container">
          <input 
            type="number" 
            placeholder="Enter time" 
            value={time} 
            onChange={(e) => setTime(e.target.value)} 
          />
          <input 
            type="number" 
            placeholder="Enter amount" 
            value={amount} 
            onChange={(e) => setAmount(e.target.value)} 
          />
          <button onClick={handleSubmit}>Check</button>
        </div>
        {result && <div className={`result ${result.includes('Fraudulent') ? 'fraud' : 'non-fraud'}`}>{result}</div>}
      </div>
    </div>
  );
}

export default App;

