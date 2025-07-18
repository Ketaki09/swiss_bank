
import React from 'react';
import { Shield, TrendingUp, Banknote } from 'lucide-react';

const Hero = () => {
  return (
    <section className="pt-20 pb-16 relative bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white overflow-hidden">
      {/* Background Image */}
      <div 
        className="absolute inset-0 bg-cover bg-center bg-no-repeat"
        style={{
          backgroundImage: `url('/Images_upload/hero-img-1-1536x819.jpeg')`
        }}
      />
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="flex items-center min-h-[60vh]">
          {/* Content - Left Side 40% */}
          <div className="w-2/5 space-y-8">
            <div className="space-y-4">
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold leading-tight font-serif">
                Swiss Excellence in
                <span className="block text-transparent bg-clip-text bg-gradient-to-r from-yellow-400 to-yellow-600">
                  Global Banking
                </span>
              </h1>
              <p className="text-lg md:text-xl text-white leading-relaxed">
                Experience unparalleled financial services backed by Swiss precision, 
                innovative technology, and over 150 years of banking expertise.
              </p>
            </div>

            <div className="flex flex-col sm:flex-row gap-4">
              <button className="bg-gradient-to-r from-yellow-500 to-yellow-600 text-slate-900 px-8 py-4 rounded-lg font-semibold hover:from-yellow-400 hover:to-yellow-500 transition-all duration-200 transform hover:scale-105">
                Open Account
              </button>
              <button className="border border-white text-white px-8 py-4 rounded-lg font-semibold hover:bg-white hover:text-slate-900 transition-all duration-200">
                Learn More
              </button> 
            </div>
            
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;

