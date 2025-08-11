// import React, { useState, useEffect, useRef } from 'react';
// import axios from 'axios';

// interface Message {
//   id: string;
//   text: string;
//   sender: 'user' | 'bot';
// }

// const App: React.FC = () => {
//   const [messages, setMessages] = useState<Message[]>([]);
//   const [input, setInput] = useState('');
//   const messagesEndRef = useRef<HTMLDivElement>(null);

//   const scrollToBottom = () => {
//     messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
//   };

//   useEffect(() => {
//     scrollToBottom();
//   }, [messages]);

//   const sendMessage = async (e: React.FormEvent) => {
//     e.preventDefault();
//     if (!input.trim()) return;

//     const userMessage: Message = { id: crypto.randomUUID(), text: input, sender: 'user' };
//     setMessages([...messages, userMessage]);
//     setInput('');

//     try {
//       const response = await axios.post('http://localhost:8000/chat', { text: input });
//       const botMessage: Message = { id: crypto.randomUUID(), text: response.data.response, sender: 'bot' };
//       setMessages((prev) => [...prev, botMessage]);
//     } catch (error) {
//       console.error('Error sending message:', error);
//       const errorMessage: Message = { id: crypto.randomUUID(), text: 'Sorry, something went wrong.', sender: 'bot' };
//       setMessages((prev) => [...prev, errorMessage]);
//     }
//   };

//   return (
//     <div className="flex flex-col h-screen bg-gray-100">
//       <div className="flex-1 overflow-y-auto p-4">
//         {messages.map((message) => (
//           <div
//             key={message.id}
//             className={`mb-4 flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
//           >
//             <div
//               className={`max-w-xs p-3 rounded-lg ${
//                 message.sender === 'user' ? 'bg-blue-500 text-white' : 'bg-white text-gray-800'
//               }`}
//             >
//               {message.text}
//             </div>
//           </div>
//         ))}
//         <div ref={messagesEndRef} />
//       </div>
//       <form onSubmit={sendMessage} className="p-4 bg-white border-t">
//         <div className="flex">
//           <input
//             type="text"
//             value={input}
//             onChange={(e) => setInput(e.target.value)}
//             className="flex-1 p-2 border rounded-l-lg focus:outline-none"
//             placeholder="Type your message..."
//           />
//           <button
//             type="submit"
//             className="p-2 bg-blue-500 text-white rounded-r-lg hover:bg-blue-600"
//           >
//             Send
//           </button>
//         </div>
//       </form>
//     </div>
//   );
// };

// export default App;

import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
// import 'animate.css';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: string;
}

// Inline SVG Icons
const HomeIcon = () => (
  <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
    <path d="M10 2L2 7v10a1 1 0 001 1h14a1 1 0 001-1V7l-8-5zm0 2.83l6 3.75V16H4V8.58l6-3.75z" />
  </svg>
);

const HistoryIcon = () => (
  <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
    <path d="M10 20a10 10 0 110-20 10 10 0 010 20zm0-2a8 8 0 100-16 8 8 0 000 16zm1-12v5l3 3-1.414 1.414-3.5-3.5V6h2z" />
  </svg>
);

const ThemeIcon = () => (
  <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
    <path fillRule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.707.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 11-2 0 1 1 0 012 0zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 11-2 0 1 1 0 012 0zm-1-1a8 8 0 1116 0 8 8 0 01-16 0z" clipRule="evenodd" />
  </svg>
);

const MenuIcon = () => (
  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
    <path fillRule="evenodd" d="M3 5h14a1 1 0 010 2H3a1 1 0 010-2zm0 4h14a1 1 0 010 2H3a1 1 0 010-2zm0 4h14a1 1 0 010 2H3a1 1 0 010-2z" clipRule="evenodd" />
  </svg>
);

const CloseIcon = () => (
  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
    <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
  </svg>
);

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: crypto.randomUUID(),
      text: "Welcome to the Symptom Checker Chatbot! I'm here to help with questions about abdominal pain in adults. Type a message to get started.",
      sender: 'bot',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ]);
  const [input, setInput] = useState('');
  const [isNavOpen, setIsNavOpen] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const userMessage: Message = { id: crypto.randomUUID(), text: input, sender: 'user', timestamp };
    setMessages([...messages, userMessage]);
    setInput('');
    setIsTyping(true);

    try {
      const response = await axios.post('http://localhost:8000/chat', { text: input });
      const botMessage: Message = {
        id: crypto.randomUUID(),
        text: response.data.response,
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setTimeout(() => {
        setMessages((prev) => [...prev, botMessage]);
        setIsTyping(false);
      }, 800); // Reduced typing delay for smoother UX
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        id: crypto.randomUUID(),
        text: 'Sorry, something went wrong.',
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setTimeout(() => {
        setMessages((prev) => [...prev, errorMessage]);
        setIsTyping(false);
      }, 800);
    }
  };

  const toggleNav = () => {
    setIsNavOpen(!isNavOpen);
  };

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
  };

  return (
    <div className={`flex flex-col h-screen ${isDarkMode ? 'bg-gray-900 text-white' : 'bg-teal-50 text-gray-900'} transition-colors duration-300`}>
      {/* Navigation Bar */}
      <nav className="flex items-center justify-between p-4 bg-gradient-to-r from-teal-500 to-purple-600 text-white shadow-lg fixed top-0 w-full z-10">
        <div className="flex items-center space-x-3">
          <img
            src="https://img.icons8.com/ios-filled/50/ffffff/stethoscope.png"
            alt="Logo"
            className="w-10 h-10 animate__animated animate__heartBeat animate__infinite animate__slower"
          />
          <div>
            <h1 className="text-2xl font-bold">Symptom Checker</h1>
            <p className="text-sm opacity-80">AI-powered health insights</p>
          </div>
        </div>
        <button onClick={toggleNav} className="p-2 focus:outline-none md:hidden">
          {isNavOpen ? <CloseIcon /> : <MenuIcon />}
        </button>
        <div className={`hidden md:flex items-center space-x-4`}>
          <button className="flex items-center p-2 hover:bg-teal-600 rounded transition-colors">
            <HomeIcon />
            Home
          </button>
          <button className="flex items-center p-2 hover:bg-teal-600 rounded transition-colors">
            <HistoryIcon />
            History
          </button>
          <button onClick={toggleTheme} className="flex items-center p-2 hover:bg-teal-600 rounded transition-colors">
            <ThemeIcon />
            {isDarkMode ? 'Light' : 'Dark'}
          </button>
        </div>
      </nav>
      {/* Mobile Navigation Menu */}
      <div
        className={`${
          isNavOpen ? 'block' : 'hidden'
        } md:hidden bg-teal-600 text-white p-4 fixed top-20 w-full shadow-md z-10 animate__animated animate__slideInDown`}
      >
        <ul className="space-y-2">
          <li>
            <button className="flex items-center w-full p-2 hover:bg-teal-700 rounded">
              <HomeIcon />
              Home
            </button>
          </li>
          <li>
            <button className="flex items-center w-full p-2 hover:bg-teal-700 rounded">
              <HistoryIcon />
              History
            </button>
          </li>
          <li>
            <button onClick={toggleTheme} className="flex items-center w-full p-2 hover:bg-teal-700 rounded">
              <ThemeIcon />
              {isDarkMode ? 'Light Mode' : 'Dark Mode'}
            </button>
          </li>
        </ul>
      </div>

      {/* Chat Area */}
      <div className={`flex-1 flex flex-col p-4 pt-24 md:pt-20 ${isDarkMode ? 'bg-gray-900' : 'bg-teal-50'} overflow-hidden transition-colors duration-300`}>
        <div className="flex-1 overflow-y-auto mb-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`mb-4 flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'} animate__animated ${message.sender === 'user' ? 'animate__fadeInRight' : 'animate__fadeInLeft'}`}
            >
              <div className="flex items-start space-x-2">
                {message.sender === 'bot' && (
                  <img
                    src="https://img.icons8.com/ios-filled/50/ffffff/bot.png"
                    alt="Bot Avatar"
                    className="w-8 h-8 rounded-full"
                  />
                )}
                <div
                  className={`max-w-xs sm:max-w-md p-4 rounded-lg shadow-lg ${
                    message.sender === 'user'
                      ? 'bg-purple-500 text-white'
                      : isDarkMode
                      ? 'bg-gray-800 text-white border border-gray-700'
                      : 'bg-white text-gray-900 border border-teal-200'
                  }`}
                >
                  <p>{message.text}</p>
                  <p className={`text-xs mt-1 ${message.sender === 'user' ? 'text-purple-200' : isDarkMode ? 'text-gray-400' : 'text-teal-600'}`}>
                    {message.timestamp}
                  </p>
                </div>
                {message.sender === 'user' && (
                  <img
                    src="https://img.icons8.com/ios-filled/50/ffffff/user.png"
                    alt="User Avatar"
                    className="w-8 h-8 rounded-full"
                  />
                )}
              </div>
            </div>
          ))}
          {isTyping && (
            <div className="flex justify-start mb-4">
              <div className="flex items-start space-x-2">
                <img
                  src="https://img.icons8.com/ios-filled/50/ffffff/bot.png"
                  alt="Bot Avatar"
                  className="w-8 h-8 rounded-full"
                />
                <div className="bg-gray-300 text-gray-900 p-4 rounded-lg shadow-lg animate__animated animate__pulse animate__infinite">
                  Typing...
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        <form
          onSubmit={sendMessage}
          className={`p-4 ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-teal-200'} border-t shadow-lg rounded-lg`}
        >
          <div className="flex space-x-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              className={`flex-1 p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 ${
                isDarkMode ? 'bg-gray-700 text-white border-gray-600' : 'bg-white text-gray-900 border-teal-300'
              }`}
              placeholder="Ask about abdominal pain..."
              aria-label="Type your message"
            />
            <button
              type="submit"
              className="p-3 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors"
              aria-label="Send message"
            >
              Send
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default App;