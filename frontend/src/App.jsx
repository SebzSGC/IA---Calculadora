import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import remarkGfm from 'remark-gfm'
import 'katex/dist/katex.min.css'
import './App.css'

const TypewriterMarkdown = ({ content, isTyping, onComplete }) => {
  const [displayedContent, setDisplayedContent] = useState(isTyping ? '' : content);

  useEffect(() => {
    if (!isTyping) {
      setDisplayedContent(content);
      return;
    }
    
    let i = 0;
    const interval = setInterval(() => {
      setDisplayedContent(content.substring(0, i));
      i += 3; // Revelar de a 3 caracteres para mayor fluidez
      if (i >= content.length) {
        clearInterval(interval);
        setDisplayedContent(content);
        if (onComplete) onComplete();
      }
    }, 10);
    
    return () => clearInterval(interval);
  }, [content, isTyping]);

  return (
    <ReactMarkdown 
      remarkPlugins={[remarkMath, remarkGfm]}
      rehypePlugins={[rehypeKatex]}
    >
      {displayedContent}
    </ReactMarkdown>
  );
};

function App() {
  const [chats, setChats] = useState(() => {
    const saved = localStorage.getItem('chatsV2');
    if (saved) {
      const parsed = JSON.parse(saved);
      if (parsed.length > 0) {
        // Al recargar, forzamos typing a false para todos los mensajes guardados
        return parsed.map(c => ({
          ...c,
          messages: c.messages.map(m => ({ ...m, typing: false }))
        }));
      }
    }
    return [{ id: Date.now().toString(), title: 'Nuevo Chat', messages: [] }];
  });

  const [currentChatId, setCurrentChatId] = useState(() => {
    const saved = localStorage.getItem('chatsV2');
    if (saved) {
      const parsed = JSON.parse(saved);
      if (parsed.length > 0) return parsed[0].id;
    }
    return chats[0]?.id || Date.now().toString();
  });

  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);
  const textareaRef = useRef(null);

  const currentChat = chats.find(c => c.id === currentChatId) || { messages: [] };
  const messages = currentChat.messages;

  // Auto-scroll al fondo
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  // Guardar historial
  useEffect(() => {
    localStorage.setItem('chatsV2', JSON.stringify(chats));
  }, [chats]);

  // Redimensionar el textarea dinámicamente
  const handleInput = (e) => {
    setInput(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 200) + 'px';
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSolve();
    }
  };

  const handleTypingComplete = (chatId, msgIndex) => {
    setChats(prev => prev.map(chat => {
      if (chat.id === chatId) {
        const newMessages = [...chat.messages];
        if (newMessages[msgIndex]) {
          newMessages[msgIndex] = { ...newMessages[msgIndex], typing: false };
        }
        return { ...chat, messages: newMessages };
      }
      return chat;
    }));
  };

  const handleSolve = async () => {
    if (!input.trim() || loading) return;
    
    const userMsg = input.trim();
    setInput('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }

    // Actualizar título si es el primer mensaje y agregar mensaje de usuario
    let updatedChats = chats.map(chat => {
      if (chat.id === currentChatId) {
        return { 
          ...chat, 
          title: chat.messages.length === 0 ? userMsg.substring(0, 30) + '...' : chat.title,
          messages: [...chat.messages, { role: 'user', content: userMsg }] 
        };
      }
      return chat;
    });
    setChats(updatedChats);
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/solve', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          problem: userMsg,
          use_cache: true
        })
      });

      if (!response.ok) {
        throw new Error('Error de conexión con el RAG de FastAPI.');
      }

      const data = await response.json();
      
      setChats(prev => prev.map(chat => {
        if (chat.id === currentChatId) {
          return { ...chat, messages: [...chat.messages, { role: 'bot', content: data.response, typing: true }] };
        }
        return chat;
      }));

    } catch (err) {
      setChats(prev => prev.map(chat => {
        if (chat.id === currentChatId) {
          return { 
            ...chat, 
            messages: [...chat.messages, { role: 'bot', content: `**Error:** ${err.message}` }] 
          };
        }
        return chat;
      }));
    } finally {
      setLoading(false);
    }
  }

  const handleNewChat = async () => {
    try {
      await fetch('http://localhost:8000/cache', { method: 'DELETE' });
    } catch (e) {
      // Ignorar error de backend
    }
    const newId = Date.now().toString();
    setChats([{ id: newId, title: 'Nuevo Chat', messages: [] }, ...chats]);
    setCurrentChatId(newId);
  }

  const handleDeleteChat = (id, e) => {
    e.stopPropagation();
    const filtered = chats.filter(c => c.id !== id);
    if (filtered.length === 0) {
      const newId = Date.now().toString();
      setChats([{ id: newId, title: 'Nuevo Chat', messages: [] }]);
      setCurrentChatId(newId);
    } else {
      setChats(filtered);
      if (currentChatId === id) setCurrentChatId(filtered[0].id);
    }
  }

  return (
    <div className="app-layout">
      {/* SIDEBAR LATERAL */}
      <div className="sidebar">
        <button className="new-chat-btn" onClick={handleNewChat}>
          + Nuevo Chat
        </button>
        <div className="chat-list">
          {chats.map(chat => (
            <div 
              key={chat.id} 
              className={`chat-item ${chat.id === currentChatId ? 'active' : ''}`}
              onClick={() => {
                if (chat.id !== currentChatId) {
                  // Apagar cualquier animación en progreso del chat actual al salir
                  setChats(prev => prev.map(c => 
                    c.id === currentChatId 
                      ? { ...c, messages: c.messages.map(m => ({ ...m, typing: false })) }
                      : c
                  ));
                  setCurrentChatId(chat.id);
                }
              }}
            >
              <div className="chat-item-title">{chat.title}</div>
              <button 
                className="delete-chat-btn" 
                title="Eliminar"
                onClick={(e) => handleDeleteChat(chat.id, e)}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* ÁREA PRINCIPAL */}
      <div className="app-container">
        <header className="header">
          <div className="title">Calculadora I.O. RAG</div>
        </header>

        <div className="chat-container">
          {messages.length === 0 ? (
            <div className="welcome-screen">
              <h1 className="welcome-title">¿En qué puedo ayudarte?</h1>
              <p className="welcome-subtitle">
                Pega aquí tu problema de Programación Lineal, Teoría de Redes o PERT/CPM. Extraeré la teoría estrictamente de las páginas de Hillier y generaré el modelo por ti.
              </p>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div key={idx} className={`message-wrapper ${msg.role}`}>
                <div className="message-content">
                  <div className={`avatar ${msg.role}`}>
                    {msg.role === 'user' ? 'U' : 'IO'}
                  </div>
                  <div className="message-text markdown-body">
                    {msg.role === 'user' ? (
                      <ReactMarkdown 
                        remarkPlugins={[remarkMath, remarkGfm]}
                        rehypePlugins={[rehypeKatex]}
                      >
                        {msg.content}
                      </ReactMarkdown>
                    ) : (
                      <TypewriterMarkdown 
                        content={msg.content} 
                        isTyping={msg.typing} 
                        onComplete={() => handleTypingComplete(chat.id, idx)}
                      />
                    )}
                  </div>
                </div>
              </div>
            ))
          )}

          {loading && (
            <div className="message-wrapper bot">
              <div className="message-content">
                <div className="avatar bot">IO</div>
                <div className="message-text">
                  <div className="typing-indicator">
                    <span></span><span></span><span></span>
                  </div>
                </div>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <div className="input-area">
          <div className="input-container">
            <textarea 
              ref={textareaRef}
              value={input}
              onChange={handleInput}
              onKeyDown={handleKeyDown}
              placeholder="Pega un problema aquí y presiona Enter..."
              rows={1}
            />
            <button 
              className="send-btn" 
              onClick={handleSolve}
              disabled={!input.trim() || loading}
            >
              ↑
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
