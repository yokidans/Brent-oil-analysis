// src/dashboard/frontend/src/components/ChangePointAnalysis.js
import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const ChangePointAnalysis = () => {
  const [changePoints, setChangePoints] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchChangePoints = async () => {
      const response = await fetch('http://localhost:5000/api/changepoints');
      const data = await response.json();
      setChangePoints(data.changepoints);
      setLoading(false);
    };
    
    fetchChangePoints();
  }, []);

  if (loading) return <div>Loading change point analysis...</div>;

  // Prepare data for visualization
  const chartData = changePoints.map(cp => ({
    date: new Date(cp.change_point_date).toLocaleDateString(),
    probability: cp.probability * 100,
    priceChange: cp.price_change_pct
  }));

  return (
    <div className="analysis-container">
      <h2>Change Point Analysis</h2>
      
      <div className="chart-row">
        <div className="chart" style={{ height: '400px', width: '50%' }}>
          <h3>Change Point Probability</h3>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis label={{ value: 'Probability (%)', angle: -90 }} />
              <Tooltip />
              <Legend />
              <Bar dataKey="probability" fill="#8884d8" name="Probability" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        
        <div className="chart" style={{ height: '400px', width: '50%' }}>
          <h3>Price Impact</h3>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis label={{ value: 'Price Change (%)', angle: -90 }} />
              <Tooltip />
              <Legend />
              <Bar dataKey="priceChange" fill="#82ca9d" name="Price Change" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      <div className="details">
        {changePoints.map((cp, idx) => (
          <div key={idx} className="change-point-card">
            <h3>Change Point on {new Date(cp.change_point_date).toLocaleDateString()}</h3>
            <p><strong>Probability:</strong> {(cp.probability * 100).toFixed(1)}%</p>
            <p><strong>Price Change:</strong> {cp.price_change_pct.toFixed(2)}%</p>
            
            {cp.nearby_events.length > 0 ? (
              <>
                <p><strong>Nearby Events:</strong></p>
                <ul>
                  {cp.nearby_events.map((event, i) => (
                    <li key={i}>
                      <strong>{new Date(event.Date).toLocaleDateString()}:</strong> {event.Description}
                    </li>
                  ))}
                </ul>
              </>
            ) : (
              <p>No nearby events identified</p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default ChangePointAnalysis;