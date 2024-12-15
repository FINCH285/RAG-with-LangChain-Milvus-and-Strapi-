import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import axios from 'axios';
import './ChatbotUI.css';

const ChatbotUI = () => {
  const [chatHistory, setChatHistory] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const chatContainerRef = useRef(null);

  useEffect(() => {
    // Scroll to the latest message when chat history updates
    chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
  }, [chatHistory]);

  const handleUserInput = (e) => setUserInput(e.target.value);

  const handleSendMessage = async () => {
    if (userInput.trim() !== '') {
      const newMessage = { role: 'user', content: userInput };
      const updatedChatHistory = [...chatHistory, newMessage];
      setChatHistory(updatedChatHistory);
      setUserInput('');
      setIsLoading(true);

      try {
        const response = await axios.post('http://localhost:30080/chat', {
          chatHistory: updatedChatHistory,
          input: userInput,
        });

        const botMessage = {
          role: 'assistant',
          content: response.data.answer || 'Sorry, I didn’t quite get that.',
        };

        setTimeout(() => {
          setChatHistory((prevMessages) => [...prevMessages, botMessage]);
        }, 1000); // Simulate typing delay
      } catch (err) {
        console.error('Error:', err);
        setError('Unable to connect to the server.');
      } finally {
        setIsLoading(false);
      }
    }
  };

  return (
    <div className="chatbot-ui">
      {/* Header */}
      <div className="chat-header">
        <img src="/milvus_logo.png" alt="Assistant" className="assistant-logo" />
        <h2>Chat Assistant</h2>
      </div>

      {/* Chat Messages */}
      <div className="chat-body" ref={chatContainerRef}>
        {chatHistory.map((message, index) => (
          <div
            key={index}
            className={`message ${message.role === 'user' ? 'user-message' : 'bot-message'}`}
          >
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        ))}
        {isLoading && (
          <div className="bot-message typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
        )}
        {error && <div className="error-message">{error}</div>}
      </div>

      {/* Footer */}
      <div className="chat-footer">
        <input
          type="text"
          placeholder="Ask me anything..."
          value={userInput}
          onChange={handleUserInput}
          onKeyPress={(e) => {
            if (e.key === 'Enter') handleSendMessage();
          }}
          disabled={isLoading}
        />
        <button onClick={handleSendMessage} disabled={isLoading}>
          <span className="send-icon">✈</span>
        </button>
      </div>
    </div>
  );
};

export default ChatbotUI;
