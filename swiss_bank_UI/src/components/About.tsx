
import React from 'react';
import { Shield, TrendingUp, Wallet, Users, Building, Award } from 'lucide-react';

const About = () => {
  const stats = [
    { label: 'Years of Excellence', value: '150+', icon: Shield },
    { label: 'Assets Under Management', value: 'CHF 50B+', icon: Wallet },
    { label: 'Global Offices', value: '25+', icon: TrendingUp }
  ];

  const keyFigures = [
    {
      value: '2,558',
      label: 'Employees at the heart of our Group',
      icon: Users
    },
    {
      value: '30',
      label: 'Offices worldwide to be close to our clients',
      icon: Building
    },
    {
      value: '224.2',
      label: 'Asset under management (CHF bn), growing steadily year on year',
      icon: TrendingUp
    },
    {
      value: '43%',
      label: 'CET1 ratio',
      icon: Award
    }
  ];

  return (
    <section id="about" className="py-20 bg-gradient-to-br from-black via-gray-900 to-black">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid lg:grid-cols-2 gap-16 items-center mb-20">
          {/* Content */}
          <div className="space-y-8">
            <div>
              <h2 className="text-3xl md:text-4xl font-bold text-white mb-6 font-serif">
                Swiss Banking Heritage, Global Innovation
              </h2>
              <div className="space-y-4 text-gray-300 leading-relaxed">
                <p>
                  Founded in 1873 in the heart of Switzerland, SwissBank International has evolved 
                  from a traditional private bank to a global financial powerhouse while maintaining 
                  our commitment to Swiss banking principles of discretion, stability, and excellence.
                </p>
                <p>
                  Our heritage of over 150 years has taught us that true wealth management goes beyond 
                  numbers. We understand that each client's financial journey is unique, requiring 
                  personalized solutions backed by institutional expertise and cutting-edge technology.
                </p>
                <p>
                  Today, we serve discerning clients across 25+ countries, managing over CHF 50 billion 
                  in assets while staying true to our founding values of integrity, innovation, and 
                  unwavering commitment to client success.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-6">
              {stats.map((stat, index) => (
                <div key={index} className="text-center">
                  <div className="bg-gradient-to-br from-gray-800 to-gray-700 p-3 rounded-xl w-12 h-12 flex items-center justify-center mx-auto mb-3">
                    <stat.icon className="w-6 h-6 text-yellow-400" />
                  </div>
                  <div className="text-2xl font-bold text-white mb-1">{stat.value}</div>
                  <div className="text-sm text-gray-400">{stat.label}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Visual Effect */}
          <div className="relative group">
          {/* Glow effect */}
          <div className="absolute inset-0 bg-gradient-to-r from-yellow-400/20 to-transparent rounded-2xl blur-3xl group-hover:from-yellow-400/30 transition-all duration-500"></div>
  
          {/* Image container */}
          <div className="relative overflow-hidden rounded-2xl border border-gray-700">
          <img 
            src="/Images_upload/currency-exchange-revolution.jpeg" 
            alt="Our Commitment" 
            className="w-full h-[280px] object-cover transition-all duration-500 group-hover:scale-110 group-hover:brightness-110"
          />
    
          {/* Subtle overlay */}
          <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent opacity-60 group-hover:opacity-40 transition-opacity duration-500"></div>
          </div>
        </div>
        </div>

        {/* Key Figures Section */}
        <div className="py-16 border-t border-gray-800">
          <div className="text-center mb-12">
            <h3 className="text-3xl md:text-4xl font-bold text-white mb-4 font-serif">
              Key Figures
            </h3>
            <p className="text-gray-300 text-lg">
              Numbers that reflect our commitment to excellence and growth
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {keyFigures.map((figure, index) => (
              <div key={index} className="text-center">
                <div className="bg-gradient-to-br from-gray-800 to-gray-700 p-4 rounded-xl w-16 h-16 flex items-center justify-center mx-auto mb-4">
                  <figure.icon className="w-8 h-8 text-yellow-400" />
                </div>
                <div className="text-2xl font-bold text-white mb-1">{figure.value}</div>
                <div className="text-gray-400 text-sm leading-relaxed">{figure.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default About;
