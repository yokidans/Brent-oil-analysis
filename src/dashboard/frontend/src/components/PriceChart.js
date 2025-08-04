// src/dashboard/frontend/src/components/PriceChart.js
import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { DateRangePicker } from 'react-date-range';
import 'react-date-range/dist/styles.css'; // Main style file
import 'react-date-range/dist/theme/default.css'; // Theme CSS file

const PriceChart = () => {
  const [priceData, setPriceData] = useState([]);
  const [dateRange, setDateRange] = useState({
    startDate: new Date(2010, 0, 1),
    endDate: new Date(),
    key: 'selection'
  });
  const [showLogReturns, setShowLogReturns] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      const from = dateRange.startDate.toISOString().split('T')[0];
      const to = dateRange.endDate.toISOString().split('T')[0];
      
      const response = await fetch(
        `http://localhost:5000/api/prices?from=${from}&to=${to}`
      );
      const data = await response.json();
      
      const formattedData = data.dates.map((date, i) => ({
        date,
        price: data.prices[i],
        logReturn: data.log_returns[i]
      }));
      
      setPriceData(formattedData);
    };
    
    fetchData();
  }, [dateRange]);

  return (
    <div className="chart-container">
      <h2>Brent Oil Price Analysis</h2>
      
      <div className="controls">
        <DateRangePicker
          ranges={[dateRange]}
          onChange={ranges => setDateRange(ranges.selection)}
        />
        
        <label>
          <input
            type="checkbox"
            checked={showLogReturns}
            onChange={() => setShowLogReturns(!showLogReturns)}
          />
          Show Log Returns
        </label>
      </div>
      
      <div className="chart" style={{ height: '500px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={priceData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip />
            <Legend />
            {showLogReturns ? (
              <Line
                type="monotone"
                dataKey="logReturn"
                stroke="#8884d8"
                activeDot={{ r: 8 }}
                name="Log Returns"
              />
            ) : (
              <Line
                type="monotone"
                dataKey="price"
                stroke="#82ca9d"
                name="Price (USD)"
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default PriceChart;