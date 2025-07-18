import { useEffect, useRef } from 'react';
import { Button } from "@/components/ui/button";
import Navigation from "@/components/Navigation";
import Hero from "@/components/Hero";
import TradingWidget from "@/components/TradingWidget";
import About from "@/components/About";
import Services from "@/components/Services";
import Footer from "@/components/Footer";
import EvaChat from "@/components/EvaChat";
import Contact from "@/components/Contact";

const Index = () => {
  return (
    <div className="min-h-screen bg-black text-white">
      <Navigation />
      <Hero />
      <Services />
      <TradingWidget />
      <About />
      <Contact />
      <Footer />
      <EvaChat />
    </div>
  );
};

export default Index;



