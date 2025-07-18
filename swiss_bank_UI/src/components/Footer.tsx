import React from 'react';
import { Shield } from 'lucide-react';

const Footer = () => {
  const footerSections = [
    {
      title: 'Company',
      links: ['About Us', 'Leadership', 'Careers', 'Investor Relations', 'Sustainability']
    },
    {
      title: 'Legal',
      links: ['Privacy Policy', 'Terms of Service', 'Regulatory Information', 'Cookie Policy', 'Compliance']
    },
    {
      title: 'Contact',
      links: ['Client Support', 'Office Locations', 'Media Relations', 'Whistleblowing', 'Feedback']
    }
  ];

  return (
    <footer className="bg-black-900 border-t border-black-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="col-span-1">
            <div className="flex items-center mb-4">
              <img src="Images_upload/bank_logo.png" alt="Bank of Swiss Logo" className="h-8 w-8" />
              <span className="text-3xl text-yellow-400 font-serif"> Bank of Swiss</span>
            </div>
            <p className="text-black-400 text-sm">
              Swiss Excellence in Global Banking since 1873
            </p>
          </div>
          
          {footerSections.map((section, index) => (
            <div key={index}>
              <h4 className="font-serif text-lg font-semibold text-white mb-4">{section.title}</h4>
              <ul className="space-y-2">
                {section.links.map((link, linkIndex) => (
                  <li key={linkIndex}>
                    <a href="#" className="text-gray-400 hover:text-yellow-400 transition-colors text-sm">
                      {link}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
        
        <div className="border-t border-gray-800 mt-8 pt-8 text-center">
          <p className="text-gray-400 text-sm">
            © 2025 Bank of Swiss. All rights reserved. Licensed and regulated by Swiss Financial Market Supervisory Authority.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;



