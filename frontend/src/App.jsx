/**
 * Main Application Component
 * 
 * The root component that orchestrates the entire RAG chat interface.
 * Manages global state for messages, documents, and loading states.
 */

import React, { useState, useCallback } from 'react';
import ChatInterface from './components/ChatInterface';
import FileUpload from './components/FileUpload';
import { askQuestion, uploadDocument, getDocuments } from './services/api';
import { FileText, MessageSquare, Database, Sparkles } from 'lucide-react';

function App() {
  // View state: 'chat' or 'upload'
  const [activeView, setActiveView] = useState('chat');
  
  // Message history
  const [messages, setMessages] = useState([]);
  
  // Loading state
  const [isLoading, setIsLoading] = useState(false);
  
  // Document stats
  const [docStats, setDocStats] = useState({ total: 0, documents: [] });
  
  // Error state
  const [error, setError] = useState(null);

  /**
   * Handle sending a question
   */
  const handleSendMessage = useCallback(async (question) => {
    // Add user message immediately
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: question,
      timestamp: new Date().toISOString(),
    };
    
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    try {
      // Call API
      const response = await askQuestion(question);
      
      // Add assistant message
      const assistantMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: response.answer,
        confidence: response.confidence,
        sources: response.sources,
        queryId: response.query_id,
        metrics: {
          retrievalTime: response.retrieval_time_ms,
          generationTime: response.generation_time_ms,
          totalTime: response.total_time_ms,
          tokensUsed: response.tokens_used,
        },
        timestamp: new Date().toISOString(),
      };
      
      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      setError(err.message || 'Failed to get response');
      
      // Add error message to chat
      const errorMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: `I encountered an error: ${err.message}. Please try again.`,
        isError: true,
        timestamp: new Date().toISOString(),
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Handle file upload
   */
  const handleFileUpload = useCallback(async (file) => {
    try {
      const response = await uploadDocument(file);
      
      // Refresh document stats
      const docs = await getDocuments();
      setDocStats({
        total: docs.total_documents,
        documents: docs.documents,
      });
      
      return response;
    } catch (err) {
      throw new Error(err.message || 'Failed to upload document');
    }
  }, []);

  /**
   * Load document stats on mount
   */
  React.useEffect(() => {
    async function loadDocs() {
      try {
        const docs = await getDocuments();
        setDocStats({
          total: docs.total_documents,
          documents: docs.documents,
        });
      } catch (err) {
        // Silently fail - API might not be running
        console.log('Could not load documents:', err.message);
      }
    }
    loadDocs();
  }, []);

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-surface-800 bg-surface-900/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            {/* Logo and title */}
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-lg font-semibold text-surface-100">
                  Enterprise RAG Assistant
                </h1>
                <p className="text-xs text-surface-500">
                  AI-powered document Q&A
                </p>
              </div>
            </div>

            {/* Navigation */}
            <nav className="flex items-center gap-2">
              <button
                onClick={() => setActiveView('chat')}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                  activeView === 'chat'
                    ? 'bg-primary-600 text-white'
                    : 'text-surface-400 hover:text-surface-200 hover:bg-surface-800'
                }`}
              >
                <MessageSquare className="w-4 h-4" />
                <span className="text-sm font-medium">Chat</span>
              </button>
              
              <button
                onClick={() => setActiveView('upload')}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                  activeView === 'upload'
                    ? 'bg-primary-600 text-white'
                    : 'text-surface-400 hover:text-surface-200 hover:bg-surface-800'
                }`}
              >
                <FileText className="w-4 h-4" />
                <span className="text-sm font-medium">Documents</span>
              </button>
            </nav>

            {/* Document count badge */}
            <div className="flex items-center gap-2 px-3 py-1.5 bg-surface-800 rounded-lg border border-surface-700">
              <Database className="w-4 h-4 text-primary-500" />
              <span className="text-sm text-surface-300">
                {docStats.total} {docStats.total === 1 ? 'document' : 'documents'}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1 max-w-6xl mx-auto w-full px-4 py-6">
        {activeView === 'chat' ? (
          <ChatInterface
            messages={messages}
            onSendMessage={handleSendMessage}
            isLoading={isLoading}
            hasDocuments={docStats.total > 0}
          />
        ) : (
          <FileUpload
            onUpload={handleFileUpload}
            documents={docStats.documents}
            onViewChat={() => setActiveView('chat')}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-surface-800 py-4">
        <div className="max-w-6xl mx-auto px-4 text-center text-xs text-surface-600">
          Enterprise RAG Assistant • Answers based only on uploaded documents
        </div>
      </footer>
    </div>
  );
}

export default App;
