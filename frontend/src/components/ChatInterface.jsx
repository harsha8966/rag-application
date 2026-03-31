/**
 * Chat Interface Component
 * 
 * The main chat UI that displays messages and handles user input.
 * Features:
 * - Message history display
 * - Input field with send button
 * - Loading state
 * - Empty state for no documents
 */

import React, { useState, useRef, useEffect } from 'react';
import MessageBubble from './MessageBubble';
import { Send, Loader2, FileQuestion, Upload } from 'lucide-react';

function ChatInterface({ messages, onSendMessage, isLoading, hasDocuments }) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  /**
   * Handle form submission
   */
  const handleSubmit = (e) => {
    e.preventDefault();
    
    const trimmedInput = input.trim();
    if (!trimmedInput || isLoading) return;
    
    onSendMessage(trimmedInput);
    setInput('');
  };

  /**
   * Handle keyboard shortcuts
   */
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // Empty state when no documents
  if (!hasDocuments) {
    return (
      <div className="h-[calc(100vh-220px)] flex flex-col items-center justify-center text-center px-4">
        <div className="w-20 h-20 rounded-2xl bg-surface-800 border border-surface-700 flex items-center justify-center mb-6">
          <Upload className="w-10 h-10 text-surface-500" />
        </div>
        <h2 className="text-xl font-semibold text-surface-200 mb-2">
          No Documents Yet
        </h2>
        <p className="text-surface-500 max-w-md mb-6">
          Upload PDF or TXT documents to start asking questions. 
          The AI will answer based only on your uploaded content.
        </p>
        <p className="text-sm text-surface-600">
          Click the "Documents" tab above to upload files
        </p>
      </div>
    );
  }

  // Empty state when no messages
  const showEmptyChat = messages.length === 0;

  return (
    <div className="h-[calc(100vh-220px)] flex flex-col">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
        {showEmptyChat ? (
          <div className="h-full flex flex-col items-center justify-center text-center">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary-500/20 to-primary-600/20 border border-primary-500/30 flex items-center justify-center mb-6">
              <FileQuestion className="w-8 h-8 text-primary-400" />
            </div>
            <h2 className="text-xl font-semibold text-surface-200 mb-2">
              Ask a Question
            </h2>
            <p className="text-surface-500 max-w-md mb-8">
              I'll search through your documents and provide answers with source citations.
              I'll only answer based on what's in your uploaded files.
            </p>
            
            {/* Example questions */}
            <div className="space-y-2">
              <p className="text-xs text-surface-600 mb-3">Try asking:</p>
              <div className="flex flex-wrap gap-2 justify-center max-w-lg">
                {[
                  "What are the main topics covered?",
                  "Summarize the key points",
                  "What policies are mentioned?",
                ].map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => setInput(suggestion)}
                    className="px-3 py-1.5 text-sm text-surface-400 bg-surface-800 
                             border border-surface-700 rounded-lg
                             hover:border-surface-600 hover:text-surface-300
                             transition-colors"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <>
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
            
            {/* Loading indicator */}
            {isLoading && (
              <div className="flex items-start gap-3 animate-fade-in">
                <div className="w-8 h-8 rounded-lg bg-surface-800 border border-surface-700 flex items-center justify-center flex-shrink-0">
                  <Loader2 className="w-4 h-4 text-primary-500 animate-spin" />
                </div>
                <div className="message-assistant px-4 py-3">
                  <div className="loading-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Input area */}
      <div className="border-t border-surface-800 p-4 bg-surface-900/50 backdrop-blur-sm">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
          <div className="flex gap-3">
            <div className="flex-1 relative">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a question about your documents..."
                rows={1}
                className="input-primary resize-none min-h-[48px] max-h-[120px] pr-12"
                disabled={isLoading}
                style={{
                  height: 'auto',
                  minHeight: '48px',
                }}
                onInput={(e) => {
                  e.target.style.height = 'auto';
                  e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
                }}
              />
            </div>
            
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="btn-primary flex items-center justify-center w-12 h-12 p-0"
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </div>
          
          <p className="text-xs text-surface-600 mt-2 text-center">
            Answers are generated only from your uploaded documents
          </p>
        </form>
      </div>
    </div>
  );
}

export default ChatInterface;
