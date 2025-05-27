import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { SendHorizonal, Bot, User } from 'lucide-react';
import './index.css'; // For custom scrollbar and animation

function Chatui() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const chatRef = useRef(null);
  const inputRef = useRef(null);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = {
      sender: 'user',
      text: input,
      time: new Date().toLocaleTimeString()
    };

    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInput('');
    setLoading(true);

    const chatHistory = updatedMessages.map(msg => msg.text);

    try {
      const response = await axios.post('http://localhost:8000/chat', {
        message: input,
        chat_history: chatHistory
      });

      const botMessage = {
        sender: 'bot',
        text: response.data.answer || "Sorry, I didn't understand that.",
        time: new Date().toLocaleTimeString()
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error(error);
      setMessages(prev => [
        ...prev,
        { sender: 'bot', text: 'Error! Try again.', time: new Date().toLocaleTimeString() }
      ]);
    }

    setLoading(false);
  };

  useEffect(() => {
    chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  return (
    <div className="min-h-screen w-full flex items-center justify-center bg-gradient-to-br from-[#a31bc2] via-[#e0b3ef] to-[#f3e8fb] p-4">
      <div className="flex flex-col w-full max-w-8xl h-[95vh] bg-white/90 rounded-2xl shadow-2xl border border-[#a31bc2]/30 overflow-hidden">
        {/* Header */}
        <header className="flex-shrink-0 w-full p-4 bg-gradient-to-r from-[#a31bc2] to-[#e0b3ef] text-xl font-bold text-center text-white shadow-md border-b border-[#a31bc2]/20">
          🎬 Movie Chat Assistant
        </header>

        {/* Chat scrollable area */}
        <main
          ref={chatRef}
          className="flex-1 overflow-y-auto px-2 sm:px-8 py-4 space-y-6 custom-scrollbar bg-transparent"
        >
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex flex-col transition-all duration-300 ${
                msg.sender === 'user' ? 'items-end' : 'items-start'
              }`}
            >
              <div className={`flex items-end ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                {msg.sender === 'bot' && <Bot className="w-5 h-5 mr-2 text-[#a31bc2]" />}
                <div
                  className={`relative max-w-[80vw] sm:max-w-lg px-4 py-3 rounded-2xl text-sm shadow-lg animate-fade-in
                    ${msg.sender === 'user'
                      ? 'bg-gradient-to-br from-[#a31bc2] to-[#e0b3ef] text-white rounded-br-none'
                      : 'bg-gradient-to-br from-white to-[#f3e8fb] border border-[#a31bc2]/20 text-[#5c217e] rounded-bl-none'
                    }`}
                >
                  <p className="break-words">{msg.text}</p>
                </div>
                {msg.sender === 'user' && <User className="w-5 h-5 ml-2 text-[#a31bc2]" />}
              </div>
              {/* Time outside chat bubble */}
              <span className={`text-[10px] mt-1 opacity-70 ${msg.sender === 'user' ? 'text-right' : 'text-left'} max-w-[80vw] sm:max-w-lg text-[#a31bc2]`}>
                {msg.time}
              </span>
            </div>
          ))}
          {loading && (
            <div className="text-sm text-[#a31bc2] animate-pulse" aria-live="polite">
              Bot is typing...
            </div>
          )}
        </main>

        {/* Footer input area pinned at bottom & centered */}
        <footer className="flex-shrink-0 w-full bg-gradient-to-r from-[#a31bc2] to-[#e0b3ef] border-t border-[#a31bc2]/20 p-4">
          <form
            className="w-full flex items-center gap-2"
            onSubmit={e => {
              e.preventDefault();
              sendMessage();
            }}
          >
            <input
              ref={inputRef}
              type="text"
              className="flex-1 border border-[#a31bc2]/40 rounded-lg px-4 py-2 outline-none focus:ring-2 focus:ring-[#a31bc2] transition-shadow bg-white/80 text-[#5c217e] placeholder-[#a31bc2]/60"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask me about movies..."
              disabled={loading}
              aria-label="Type your message"
            />
            <button
              type="submit"
              disabled={loading}
              className="bg-[#a31bc2] hover:bg-[#8611a8] focus:ring-2 focus:ring-[#a31bc2] text-white p-2 rounded-full disabled:opacity-50 transition-all"
              aria-label="Send message"
              role="button"
            >
              <SendHorizonal size={20} />
            </button>
          </form>
        </footer>
      </div>
    </div>
  );
}

export default Chatui;
