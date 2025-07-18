import React, { useEffect, useRef } from 'react';
import { Button } from "@/components/ui/button";
import { TrendingUp, Landmark, Crown, Lock, DollarSign } from "lucide-react";

const Services = () => {
  const servicesRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('animate-fade-in');
          }
        });
      },
      { threshold: 0.1 }
    );

    const serviceBlocks = document.querySelectorAll('.service-block');
    serviceBlocks.forEach((block) => observer.observe(block));

    return () => observer.disconnect();
  }, []);

  const services = [
    {
      title: "Wealth Management",
      description: "Comprehensive portfolio management and investment strategies tailored to your financial goals. Our expert advisors provide personalized guidance to maximize your wealth potential.",
      offerings: ["Portfolio Management", "Investment Advisory", "Risk Assessment", "Market Analysis"],
      icon: TrendingUp,
      image: "/Images_upload/map-lying-wooden-table.jpg"
    },
    {
      title: "Corporate Banking",
      description: "Advanced banking solutions for businesses of all sizes. From cash management to international trade finance, we support your business growth.",
      offerings: ["Trade Finance", "Cash Management", "Corporate Loans", "Treasury Services"],
      icon: Landmark,
      image: "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=600&h=400&fit=crop"
    },
    {
      title: "Private Banking",
      description: "Exclusive banking services for high-net-worth individuals. Experience personalized attention with dedicated relationship managers and bespoke financial solutions.",
      offerings: ["Personal Wealth Advisory", "Estate Planning", "Tax Optimization", "Exclusive Investment Opportunities"],
      icon: Crown,
      image: "/Images_upload/Private_banking.jpeg"
    },
    {
      title: "Asset Management",
      description: "Institutional-grade investment solutions with access to global markets and alternative investments.",
      offerings: ["Institutional Funds", "Alternative Investments", "ESG Solutions", "Multi-Asset Strategies"],
      icon: TrendingUp,
      image: "/Images_upload/Asset_managment.jpeg"
    },
    {
      title: "Trading Services",
      description: "Advanced trading platforms with direct market access and institutional-grade execution.",
      offerings: ["Multi-Asset Trading", "Prime Brokerage", "Market Research", "Execution Services"],
      icon: DollarSign,
      image: "/Images_upload/Trading_services.jpg"
    },
    {
      title: "Digital Security",
      description: "State-of-the-art security measures protecting your assets and data. Our advanced encryption and multi-factor authentication ensure complete peace of mind.",
      offerings: ["Advanced Encryption", "Biometric Authentication", "24/7 Monitoring", "Fraud Protection"],
      icon: Lock,
      image: "/Images_upload/Digital_security.jpg"
    }
  ];

  return (
    <section id="services" className="py-20 bg-black" ref={servicesRef}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="font-serif text-4xl md:text-5xl font-bold text-white mb-4">
            Our Services
          </h2>
          <p className="text-xl text-gray-400 max-w-3xl mx-auto">
            Comprehensive financial solutions tailored to meet your unique needs and aspirations.
          </p>
        </div>

        <div className="space-y-20">
          {services.map((service, index) => (
            <div
              key={service.title}
              className={`service-block opacity-0 transition-all duration-1000 ${
                index % 2 === 0 ? 'slide-in-left' : 'slide-in-right'
              }`}
            >
              <div className={`flex flex-col lg:flex-row items-center gap-12 ${
                index % 2 !== 0 ? 'lg:flex-row-reverse' : ''
              }`}>
                {/* Image */}
                <div className="lg:w-1/2">
                  <div className="relative overflow-hidden rounded-2xl shadow-2xl group">
                    <img
                      src={service.image}
                      alt={service.title}
                      className="w-full h-80 object-cover transition-transform duration-500 group-hover:scale-105"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                  </div>
                </div>

                {/* Content */}
                <div className="lg:w-1/2 space-y-6">
                  <div className="flex items-center space-x-4">
                    <div className="p-3 bg-yellow-400/10 rounded-full">
                      <service.icon className="h-8 w-8 text-yellow-400" />
                    </div>
                    <h3 className="font-serif text-3xl font-bold text-white">
                      {service.title}
                    </h3>
                  </div>
                  
                  <p className="text-lg text-gray-300 leading-relaxed">
                    {service.description}
                  </p>

                  <div className="space-y-3">
                    <h4 className="font-serif text-xl font-semibold text-white">Key Offerings:</h4>
                    <ul className="space-y-2">
                      {service.offerings.map((offering, idx) => (
                        <li key={idx} className="flex items-center text-gray-300">
                          <div className="w-2 h-2 bg-yellow-400 rounded-full mr-3"></div>
                          {offering}
                        </li>
                      ))}
                    </ul>
                  </div>

                  <Button className="bg-yellow-400 text-black hover:bg-yellow-500 font-medium">
                    Learn More
                  </Button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Services;




