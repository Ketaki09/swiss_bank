
import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown } from 'lucide-react';

const TradingWidget = () => {
  const [marketData, setMarketData] = useState([
    { symbol: 'SMI', price: 11450.23, change: 0.85, changePercent: 0.0074 },
    { symbol: 'DAX', price: 15832.45, change: -23.67, changePercent: -0.0015 },
    { symbol: 'S&P 500', price: 4387.16, change: 12.45, changePercent: 0.0028 },
    { symbol: 'NIKKEI', price: 32891.03, change: 156.78, changePercent: 0.0048 },
    { symbol: 'FTSE 100', price: 7589.23, change: -8.94, changePercent: -0.0012 },
    { symbol: 'EUR/CHF', price: 0.9745, change: 0.0023, changePercent: 0.0024 }
  ]);

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setMarketData(prevData => 
        prevData.map(item => {
          const randomChange = (Math.random() - 0.5) * 0.02;
          const newPrice = item.price * (1 + randomChange);
          const change = newPrice - item.price;
          const changePercent = change / item.price;
          
          return {
            ...item,
            price: parseFloat(newPrice.toFixed(2)),
            change: parseFloat(change.toFixed(2)),
            changePercent: parseFloat(changePercent.toFixed(4))
          };
        })
      );
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  return (
    <section id="trading" className="py-20 bg-black text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold mb-4 font-serif">
            Real-Time Market Data
          </h2>
          <p className="text-gray-300 text-lg">
            Access live market data and execute trades with institutional-grade platforms
          </p>
        </div>

        <div className="bg-gray-900/50 backdrop-blur-sm rounded-2xl p-6 border border-gray-700">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-semibold font-serif">Global Markets</h3>
            <div className="flex items-center space-x-2 text-sm text-gray-400">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span>Live</span>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {marketData.map((item, index) => (
              <div
                key={index}
                className="bg-gray-800/50 p-4 rounded-lg border border-gray-600 hover:border-gray-500 transition-colors duration-200"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-semibold text-gray-200">{item.symbol}</span>
                  <div className="flex items-center">
                    {item.change >= 0 ? (
                      <TrendingUp className="w-4 h-4 text-green-400 mr-1" />
                    ) : (
                      <TrendingDown className="w-4 h-4 text-red-400 mr-1" />
                    )}
                  </div>
                </div>
                
                <div className="space-y-1">
                  <div className="text-lg font-bold">{item.price.toLocaleString()}</div>
                  <div className={`text-sm flex items-center space-x-2 ${
                    item.change >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    <span>{item.change >= 0 ? '+' : ''}{item.change}</span>
                    <span>({item.changePercent >= 0 ? '+' : ''}{(item.changePercent * 100).toFixed(2)}%)</span>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-8 text-center">
            <button className="bg-gradient-to-r from-yellow-500 to-yellow-600 text-black px-8 py-3 rounded-lg font-semibold hover:from-yellow-400 hover:to-yellow-500 transition-all duration-200 transform hover:scale-105">
              Access Trading Platform
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default TradingWidget;
