/**
 * Message Bubble Component
 * 
 * Displays individual chat messages with:
 * - User/assistant styling
 * - Confidence indicator for assistant messages
 * - Source citations
 * - Feedback buttons
 */

import React, { useState } from 'react';
import ConfidenceBar from './ConfidenceBar';
import SourcesList from './SourcesList';
import FeedbackButtons from './FeedbackButtons';
import { User, Bot, AlertCircle } from 'lucide-react';

function MessageBubble({ message }) {
  const isUser = message.role === 'user';
  const isError = message.isError;
  
  return (
    <div
      className={`flex gap-3 animate-slide-up ${
        isUser ? 'flex-row-reverse' : 'flex-row'
      }`}
    >
      {/* Avatar */}
      <div
        className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${
          isUser
            ? 'bg-primary-600'
            : isError
            ? 'bg-rose-600/20 border border-rose-500/30'
            : 'bg-surface-800 border border-surface-700'
        }`}
      >
        {isUser ? (
          <User className="w-4 h-4 text-white" />
        ) : isError ? (
          <AlertCircle className="w-4 h-4 text-rose-400" />
        ) : (
          <Bot className="w-4 h-4 text-primary-400" />
        )}
      </div>

      {/* Message content */}
      <div className={`flex flex-col gap-2 max-w-[80%] ${isUser ? 'items-end' : 'items-start'}`}>
        {/* Main message bubble */}
        <div
          className={`px-4 py-3 ${
            isUser
              ? 'message-user'
              : isError
              ? 'bg-rose-500/10 border border-rose-500/20 rounded-2xl rounded-bl-md text-rose-200'
              : 'message-assistant'
          }`}
        >
          <div className="prose-dark whitespace-pre-wrap">
            {message.content}
          </div>
        </div>

        {/* Assistant-only extras */}
        {!isUser && !isError && (
          <div className="space-y-3 w-full">
            {/* Confidence indicator */}
            {message.confidence && (
              <ConfidenceBar confidence={message.confidence} />
            )}

            {/* Source citations */}
            {message.sources && message.sources.length > 0 && (
              <SourcesList sources={message.sources} />
            )}

            {/* Feedback buttons */}
            {message.queryId && (
              <FeedbackButtons queryId={message.queryId} />
            )}

            {/* Performance metrics (collapsed by default) */}
            {message.metrics && (
              <MetricsDisplay metrics={message.metrics} />
            )}
          </div>
        )}

        {/* Timestamp */}
        <span className="text-xs text-surface-600">
          {new Date(message.timestamp).toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
          })}
        </span>
      </div>
    </div>
  );
}

/**
 * Collapsible metrics display
 */
function MetricsDisplay({ metrics }) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="text-xs text-surface-600">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="hover:text-surface-400 transition-colors"
      >
        {isExpanded ? '▼' : '▶'} Performance details
      </button>
      
      {isExpanded && (
        <div className="mt-2 p-2 bg-surface-800/50 rounded-lg space-y-1 font-mono">
          <div>Retrieval: {metrics.retrievalTime?.toFixed(0)}ms</div>
          <div>Generation: {metrics.generationTime?.toFixed(0)}ms</div>
          <div>Total: {metrics.totalTime?.toFixed(0)}ms</div>
          {metrics.tokensUsed && (
            <div>
              Tokens: {metrics.tokensUsed.prompt} prompt + {metrics.tokensUsed.completion} completion
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default MessageBubble;
