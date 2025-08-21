// Path: swiss_bank_UI/src/components/agent/WelcomeScreen.tsx

import React from 'react';

interface WelcomeScreenProps {
  onSendMessage: (message: string) => void;
}

const WelcomeScreen: React.FC<WelcomeScreenProps> = ({ onSendMessage }) => {
  const quickActions = [
    {
      text: "Banking Policy Guidelines",
      prompt: "What are the current banking policy guidelines for customer onboarding?"
    },
    {
      text: "Compliance Requirements",
      prompt: "What are the latest compliance requirements for financial transactions?"
    },
    {
      text: "Risk Assessment",
      prompt: "Explain the risk assessment framework for new customers"
    }
  ];

  const handleQuickAction = (prompt: string) => {
    onSendMessage(prompt);
  };

  return (
    <div className="flex flex-col items-center w-full h-full justify-center font-serif bg-black">
      {/* Main Content Container */}
      <div className="flex flex-col items-center gap-8 w-full max-w-4xl px-4">
        
        {/* Logo Section */}
        <div className="flex flex-col items-center justify-center gap-6">
          {/* Swiss Bank Logo - Larger */}
          <div className="w-20 h-20 bg-black rounded-full flex items-center justify-center overflow-hidden border-2 border-yellow-400">
            <img 
              src="/Images_upload/bank_logo.png" 
              alt="Swiss Bank Logo" 
              className="w-16 h-16 object-contain"
            />
          </div>
          
          {/* Welcome Text */}
          <div className="text-center">
            <h1 className="text-2xl font-bold text-yellow-400 mb-2 font-serif">
              Swiss Agent
            </h1>
            <p className="text-gray-300 text-lg font-serif">
              Your Internal Banking Assistant
            </p>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="w-full flex justify-center">
          <div className="w-full max-w-3xl">
            <div className="flex flex-row flex-wrap w-full gap-3 justify-center items-center">
              {quickActions.map((action, index) => (
                <button
                  key={index}
                  onClick={() => handleQuickAction(action.prompt)}
                  className="inline-flex items-center justify-center gap-2 whitespace-nowrap text-sm font-medium leading-normal cursor-pointer focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-yellow-400 disabled:opacity-60 disabled:cursor-not-allowed transition-all duration-200 select-none border border-yellow-400 text-white hover:bg-yellow-500 hover:text-black hover:scale-105 active:scale-95 disabled:hover:bg-transparent h-12 px-6 py-3 rounded-full font-serif"
                >
                  <svg 
                    width="16" 
                    height="16" 
                    viewBox="0 0 24 24" 
                    fill="none" 
                    xmlns="http://www.w3.org/2000/svg" 
                    className="stroke-2 text-yellow-400 group-hover:text-black transition-colors"
                  >
                    <path 
                      d="M12 2L13.09 8.26L22 12L13.09 15.74L12 22L10.91 15.74L2 12L10.91 8.26L12 2Z" 
                      stroke="currentColor" 
                      strokeLinecap="round" 
                      strokeLinejoin="round"
                    />
                  </svg>
                  <span className="overflow-hidden whitespace-nowrap text-ellipsis">
                    {action.text}
                  </span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WelcomeScreen;