/**
 * Feedback Buttons Component
 * 
 * Allows users to provide feedback on answers:
 * - Thumbs up (positive)
 * - Thumbs down (negative)
 * 
 * Feedback helps improve the system without retraining.
 */

import React, { useState } from 'react';
import { ThumbsUp, ThumbsDown, Check, Loader2 } from 'lucide-react';
import { submitFeedback } from '../services/api';

function FeedbackButtons({ queryId }) {
  const [feedbackState, setFeedbackState] = useState('none'); // 'none', 'loading', 'positive', 'negative'
  const [error, setError] = useState(null);

  const handleFeedback = async (type) => {
    if (feedbackState !== 'none') return;
    
    setFeedbackState('loading');
    setError(null);

    try {
      await submitFeedback(queryId, type);
      setFeedbackState(type);
    } catch (err) {
      setError('Failed to submit feedback');
      setFeedbackState('none');
    }
  };

  // Already submitted feedback
  if (feedbackState === 'positive' || feedbackState === 'negative') {
    return (
      <div className="flex items-center gap-2 text-xs text-surface-500">
        <Check className="w-3 h-3 text-emerald-500" />
        <span>
          Thanks for your feedback!
        </span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-surface-600">Was this helpful?</span>
      
      <div className="flex items-center gap-1">
        {/* Thumbs up */}
        <button
          onClick={() => handleFeedback('positive')}
          disabled={feedbackState === 'loading'}
          className="btn-icon group"
          title="Yes, helpful"
        >
          {feedbackState === 'loading' ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <ThumbsUp className="w-4 h-4 group-hover:text-emerald-400 transition-colors" />
          )}
        </button>

        {/* Thumbs down */}
        <button
          onClick={() => handleFeedback('negative')}
          disabled={feedbackState === 'loading'}
          className="btn-icon group"
          title="No, not helpful"
        >
          {feedbackState === 'loading' ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <ThumbsDown className="w-4 h-4 group-hover:text-rose-400 transition-colors" />
          )}
        </button>
      </div>

      {error && (
        <span className="text-xs text-rose-400">{error}</span>
      )}
    </div>
  );
}

export default FeedbackButtons;
