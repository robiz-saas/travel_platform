
import { useState, useEffect } from 'react'
import { MessageCircle, Send, X, Bot, User } from 'lucide-react'
import './App.css'

function App() {
  const [messages, setMessages] = useState([
    { id: 1, text: "Hello! I'm your AI assistant. How can I help you today?", sender: 'bot' }
  ])
  const [inputMessage, setInputMessage] = useState('')
  const [isOpen, setIsOpen] = useState(false)
  const [isTyping, setIsTyping] = useState(false)
  const [isBackendReady, setIsBackendReady] = useState(false)

  // Check if backend is ready when component mounts
  useEffect(() => {
  checkBackendHealth()
}, [])


  const checkBackendHealth = async () => {
    try {
      const response = await fetch('http://localhost:8000/health', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      })
      
      if (response.ok) {
        setIsBackendReady(true)
        console.log('Backend is ready!')
      } else {
        console.log('Backend health check failed')
      }
    } catch (error) {
      console.log('Backend not available:', error)
      setIsBackendReady(false)
    }
  }

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return

    const userMessage = {
      id: messages.length + 1,
      text: inputMessage,
      sender: 'user'
    }

    setMessages(prev => [...prev, userMessage])
    const currentInput = inputMessage
    setInputMessage('')
    setIsTyping(true)

    try {
      // Call your FastAPI backend
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: currentInput,
          conversation_id: 'web_chat_' + Date.now() // Optional: for conversation tracking
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      
      const botResponse = {
        id: messages.length + 2,
        text: data.response || data.message || 'Sorry, I encountered an error processing your request.',
        sender: 'bot'
      }
      
      setMessages(prev => [...prev, botResponse])
    } catch (error) {
      console.error('Error calling backend:', error)
      
      const errorResponse = {
        id: messages.length + 2,
        text: isBackendReady 
          ? 'Sorry, I encountered an error. Please try again.' 
          : 'Backend is not available. Please make sure your FastAPI server is running on http://localhost:8000',
        sender: 'bot'
      }
      
      setMessages(prev => [...prev, errorResponse])
    } finally {
      setIsTyping(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  return (
    <div className="app">
      <div className="main-content">
        <h1>AI Chat Assistant</h1>
        <p>Click the chat button to start a conversation with your custom AI assistant!</p>
        <div className="backend-status">
          <span className={`status-indicator ${isBackendReady ? 'ready' : 'not-ready'}`}>
            {isBackendReady ? '🟢 Backend Ready' : '🔴 Backend Not Available'}
          </span>
        </div>
      </div>

      {/* Chat Button */}
      <button 
        className="chat-toggle"
        onClick={() => setIsOpen(!isOpen)}
      >
        {isOpen ? <X size={24} /> : <MessageCircle size={24} />}
      </button>

      {/* Chat Window */}
      {isOpen && (
        <div className="chat-window">
          <div className="chat-header">
            <div className="header-info">
              <Bot size={20} />
              <span>AI Assistant</span>
              <span className={`connection-dot ${isBackendReady ? 'connected' : 'disconnected'}`}></span>
            </div>
            <button onClick={() => setIsOpen(false)}>
              <X size={20} />
            </button>
          </div>

          <div className="chat-messages">
            {messages.map((message) => (
              <div key={message.id} className={`message ${message.sender}`}>
                <div className="message-avatar">
                  {message.sender === 'bot' ? <Bot size={16} /> : <User size={16} />}
                </div>
                <div className="message-content">
                  <p>{message.text}</p>
                </div>
              </div>
            ))}
            
            {isTyping && (
              <div className="message bot">
                <div className="message-avatar">
                  <Bot size={16} />
                </div>
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="chat-input">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={isBackendReady ? "Type your message..." : "Backend not available..."}
              rows="1"
              disabled={!isBackendReady}
            />
            <button 
              onClick={handleSendMessage} 
              disabled={!inputMessage.trim() || !isBackendReady}
            >
              <Send size={20} />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
