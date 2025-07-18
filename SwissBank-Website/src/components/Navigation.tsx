
import React, { useState } from 'react';
import { Menu, X, Shield } from 'lucide-react';

const Navigation = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const navItems = [
    { name: 'Private Banking', href: '#services' },
    { name: 'Corporate Banking', href: '#services' },
    { name: 'Trading', href: '#trading' },
    { name: 'About', href: '#about' },
    { name: 'Contact', href: '#contact' }
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-black/80 backdrop-blur-md border-b border-gray-700/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo Section - Fixed positioning */}
          <div className="flex items-center space-x-2">
            <img src="Images_upload/bank_logo.png" alt="Bank of Swiss Logo" className="h-8 w-8" />
            <span className="text-3xl text-yellow-400 font-serif">Bank of Swiss</span>
          </div>

          <div className="hidden md:flex items-center space-x-8">
            {navItems.map((item) => (
              <a
                key={item.name}
                href={item.href}
                className="text-white/80 hover:text-yellow-400 font-serif transition-colors duration-200 font-medium"
              >
                {item.name}
              </a>
            ))}
            <button className="bg-transparent border-2 border-white text-white px-6 py-2 rounded-full hover:bg-white/10 hover:border-yellow-400 hover:text-yellow-400 transition-all duration-300 font-serif backdrop-blur-sm">
              Login
            </button>
          </div>

          <div className="md:hidden">
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="text-white/80 hover:text-white"
            >
              {isMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
            </button>
          </div>
        </div>

        {isMenuOpen && (
          <div className="md:hidden bg-black/90 backdrop-blur-md border-t border-gray-700/50">
            <div className="px-2 pt-2 pb-3 space-y-1">
              {navItems.map((item) => (
                <a
                  key={item.name}
                  href={item.href}
                  className="block px-3 py-2 text-white/80 hover:text-white hover:text-yellow-400 transition-colors duration-200"
                  onClick={() => setIsMenuOpen(false)}
                >
                  {item.name}
                </a>
              ))}
              <button className="w-full text-left px-3 py-2 bg-yellow-500/90 text-slate-900 rounded-lg hover:bg-yellow-400 transition-colors duration-200 mt-2 font-serif">
                Login
              </button>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
};

export default Navigation;
