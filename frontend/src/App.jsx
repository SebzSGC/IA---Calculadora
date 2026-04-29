import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import remarkGfm from 'remark-gfm'
import mermaid from 'mermaid'
import 'katex/dist/katex.min.css'
import './App.css'

mermaid.initialize({
  startOnLoad: false,
  theme: 'dark',
  securityLevel: 'loose',
});

const Mermaid = ({ chart }) => {
  const [svg, setSvg] = useState('');
  
  useEffect(() => {
    let isMounted = true;
    const renderChart = async () => {
      try {
        const id = `mermaid-${Math.random().toString(36).substr(2, 9)}`;
        const { svg } = await mermaid.render(id, chart);
        if (isMounted) setSvg(svg);
      } catch (error) {
        console.error('Mermaid render error:', error);
      }
    };
    if (chart) renderChart();
    return () => { isMounted = false; };
  }, [chart]);

  return <div className="mermaid-wrapper" dangerouslySetInnerHTML={{ __html: svg }} />;
};

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

  const components = {
    code({node, inline, className, children, ...props}) {
      const match = /language-(\w+)/.exec(className || '');
      const isMermaid = match && match[1] === 'mermaid';
      
      if (!inline && isMermaid) {
        return <Mermaid chart={String(children).replace(/\n$/, '')} />;
      }
      return (
        <code className={className} {...props}>
          {children}
        </code>
      );
    }
  };

  return (
    <ReactMarkdown 
      remarkPlugins={[remarkMath, remarkGfm]}
      rehypePlugins={[rehypeKatex]}
      components={components}
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
    return chats[0]?.id;
  });

  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [dbStatus, setDbStatus] = useState('ready');
  const [useCache, setUseCache] = useState(true);
  const [isClearingCache, setIsClearingCache] = useState(false);
  const [editingChatId, setEditingChatId] = useState(null);
  const [editingTitle, setEditingTitle] = useState('');
  const [ragExplorerData, setRagExplorerData] = useState(null); // Para el modal del RAG Explorer
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

  // Hacer poll al estado de la DB cada 5 segundos si se está construyendo
  useEffect(() => {
    let intervalId;
    if (dbStatus === 'building') {
      intervalId = setInterval(async () => {
        try {
          const res = await fetch('http://localhost:8000/db-status');
          const data = await res.json();
          setDbStatus(data.status);
          if (data.status === 'ready') {
            alert('¡Base de conocimientos reconstruida exitosamente! Ya puedes preguntar.');
          }
        } catch (e) {
          console.error(e);
        }
      }, 5000);
    }
    return () => clearInterval(intervalId);
  }, [dbStatus]);

  useEffect(() => {
    // Al cargar la app por primera vez, ver el estado
    fetch('http://localhost:8000/db-status')
      .then(res => res.json())
      .then(data => setDbStatus(data.status))
      .catch(console.error);
  }, []);

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
    if (!input.trim() || loading || dbStatus === 'building') return;
    
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
          use_cache: useCache
        })
      });

      if (!response.ok) {
        throw new Error('Error de conexión con el RAG de FastAPI.');
      }

      const data = await response.json();
      
      setChats(prev => prev.map(chat => {
        if (chat.id === currentChatId) {
          return { 
            ...chat, 
            messages: [...chat.messages, { 
              role: 'bot', 
              content: data.response, 
              typing: true,
              pages: data.pages || [],
              chunks: data.chunks || []
            }] 
          };
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

  const handleNewChat = () => {
    const newId = Date.now().toString();
    setChats([{ id: newId, title: 'Nuevo Chat', messages: [] }, ...chats]);
    setCurrentChatId(newId);
  }

  const handleClearCache = async () => {
    if (!window.confirm("¿Estás seguro de borrar toda la memoria semántica? La IA volverá a gastar tokens en las mismas consultas.")) return;
    setIsClearingCache(true);
    try {
      await fetch('http://localhost:8000/cache', { method: 'DELETE' });
      alert('Memoria Semántica limpiada correctamente.');
    } catch (e) {
      alert('Error limpiando memoria: ' + e.message);
    } finally {
      setIsClearingCache(false);
    }
  }

  const handleRebuildDB = async () => {
    if (!window.confirm("ATENCIÓN: Esto eliminará la base vectorial actual y la reconstruirá desde cero. Tomará unos minutos. ¿Proceder?")) return;
    try {
      const res = await fetch('http://localhost:8000/rebuild-db', { method: 'POST' });
      if (res.ok) {
        setDbStatus('building');
      } else {
        const error = await res.json();
        alert(error.detail);
      }
    } catch (e) {
      alert('Error iniciando reconstrucción: ' + e.message);
    }
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

  const handleEditClick = (chatId, currentTitle, e) => {
    e.stopPropagation();
    setEditingChatId(chatId);
    setEditingTitle(currentTitle);
  };

  const handleSaveTitle = (chatId) => {
    if (editingTitle.trim() !== '') {
      setChats(prev => prev.map(c => 
        c.id === chatId ? { ...c, title: editingTitle.trim() } : c
      ));
    }
    setEditingChatId(null);
  };

  const handleKeyDownTitle = (e, chatId) => {
    if (e.key === 'Enter') {
      handleSaveTitle(chatId);
    } else if (e.key === 'Escape') {
      setEditingChatId(null);
    }
  };

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
              {editingChatId === chat.id ? (
                <input
                  type="text"
                  className="chat-title-input"
                  value={editingTitle}
                  onChange={(e) => setEditingTitle(e.target.value)}
                  onBlur={() => handleSaveTitle(chat.id)}
                  onKeyDown={(e) => handleKeyDownTitle(e, chat.id)}
                  autoFocus
                  onClick={(e) => e.stopPropagation()}
                />
              ) : (
                <div className="chat-item-title" onDoubleClick={(e) => handleEditClick(chat.id, chat.title, e)}>
                  {chat.title}
                </div>
              )}
              
              <div className="chat-item-actions">
                {editingChatId !== chat.id && (
                  <button 
                    className="action-btn" 
                    title="Editar nombre"
                    onClick={(e) => handleEditClick(chat.id, chat.title, e)}
                  >
                    ✎
                  </button>
                )}
                <button 
                  className="action-btn delete-chat-btn" 
                  title="Eliminar chat"
                  onClick={(e) => handleDeleteChat(chat.id, e)}
                >
                  ×
                </button>
              </div>
            </div>
          ))}
        </div>
        <div className="sidebar-footer">
          <button 
            className="clear-cache-btn" 
            onClick={handleRebuildDB}
            disabled={dbStatus === 'building'}
            title="Borrar y reconstruir la base vectorial (ChromaDB)"
            style={{ marginBottom: '0.5rem', color: 'var(--text-main)' }}
          >
            {dbStatus === 'building' ? '⚙️ Reconstruyendo (5-10 min)...' : '⚙️ Reconstruir DB Completa'}
          </button>
          <button 
            className="clear-cache-btn" 
            onClick={handleClearCache}
            disabled={isClearingCache || dbStatus === 'building'}
            title="Borrar memoria caché semántica de consultas"
          >
            {isClearingCache ? 'Borrando...' : '🗑️ Limpiar Historial Caché'}
          </button>
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
                    
                    {msg.role === 'bot' && !msg.typing && msg.chunks && msg.chunks.length > 0 && (
                      <div className="rag-metadata-panel">
                        <div className="rag-pages-badge">
                          📚 Páginas consultadas: {msg.pages.join(', ')}
                        </div>
                        <button 
                          className="rag-explorer-btn"
                          onClick={() => setRagExplorerData({ pages: msg.pages, chunks: msg.chunks })}
                        >
                          🔍 Explorador RAG Visual
                        </button>
                      </div>
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
          <div className="input-options">
            <label className="toggle-cache-label" title="Si se desactiva, Gemini procesará el texto desde cero gastando tokens.">
              <input 
                type="checkbox" 
                checked={useCache}
                onChange={(e) => setUseCache(e.target.checked)}
              />
              <span>Caché Semántico (Ahorro tokens)</span>
            </label>
          </div>
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
              disabled={!input.trim() || loading || dbStatus === 'building'}
            >
              ↑
            </button>
          </div>
          
          {dbStatus === 'building' && (
            <div className="db-building-status">
              <span className="spinner">🔄</span>
              La Inteligencia Artificial está estudiando el libro y procesando páginas (aprox. 5-10 min)...
            </div>
          )}
        </div>
      </div>

      {/* RAG EXPLORER MODAL */}
      {ragExplorerData && (
        <div className="rag-modal-overlay" onClick={() => setRagExplorerData(null)}>
          <div className="rag-modal-content" onClick={e => e.stopPropagation()}>
            <div className="rag-modal-header">
              <h2>🔍 Explorador RAG Interno</h2>
              <button className="close-modal-btn" onClick={() => setRagExplorerData(null)}>×</button>
            </div>
            
            <div className="rag-modal-body">
              <div className="rag-chunks-section">
                <h3>📝 Fragmentos Recuperados (Top {ragExplorerData.chunks.length})</h3>
                <p className="rag-subtitle">Estos son los pedazos exactos del libro que el Cross-Encoder calificó como perfectos para responder a tu duda:</p>
                
                <div className="chunks-list">
                  {ragExplorerData.chunks.map((chunk, idx) => (
                    <div key={idx} className="chunk-card">
                      <div className="chunk-header">
                        <span className="chunk-badge">Página {chunk.page}</span>
                        <span className="chunk-score" title="Cross-Encoder Re-Ranking Score">Score: {chunk.score.toFixed(2)}</span>
                      </div>
                      <div className="chunk-text">{chunk.content}</div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rag-pdf-section">
                <h3>📕 PDF de Origen (Hillier 10ª Edición)</h3>
                <iframe 
                  src={`http://localhost:8000/pdf/Investigacion-Operaciones10Edicion-Frederick-S-Hillier.pdf#page=${ragExplorerData.pages[0] || 1}`} 
                  title="PDF Viewer" 
                  className="pdf-viewer"
                ></iframe>
              </div>
            </div>
          </div>
        </div>
      )}

    </div>
  )
}

export default App
