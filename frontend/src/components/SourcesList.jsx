/**
 * Sources List Component
 * 
 * Displays the source documents used to generate an answer.
 * Shows relevance scores and allows expanding to see preview.
 */

import React, { useState } from 'react';
import { FileText, ChevronDown, ChevronUp, ExternalLink } from 'lucide-react';

function SourcesList({ sources }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [expandedSource, setExpandedSource] = useState(null);

  if (!sources || sources.length === 0) {
    return null;
  }

  // Show first 2 sources collapsed, rest on expand
  const visibleSources = isExpanded ? sources : sources.slice(0, 2);
  const hasMore = sources.length > 2;

  return (
    <div className="space-y-2">
      {/* Header */}
      <div className="flex items-center gap-2 text-sm text-surface-500">
        <FileText className="w-4 h-4" />
        <span>Sources ({sources.length})</span>
      </div>

      {/* Source chips */}
      <div className="flex flex-wrap gap-2">
        {visibleSources.map((source, index) => (
          <SourceChip
            key={`${source.source}-${source.page}-${index}`}
            source={source}
            isExpanded={expandedSource === index}
            onToggle={() => setExpandedSource(expandedSource === index ? null : index)}
          />
        ))}
        
        {/* Show more/less button */}
        {hasMore && (
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="source-chip text-primary-400 hover:text-primary-300"
          >
            {isExpanded ? (
              <>
                <ChevronUp className="w-3 h-3" />
                Show less
              </>
            ) : (
              <>
                <ChevronDown className="w-3 h-3" />
                +{sources.length - 2} more
              </>
            )}
          </button>
        )}
      </div>

      {/* Expanded source preview */}
      {expandedSource !== null && visibleSources[expandedSource] && (
        <div className="p-3 bg-surface-800/50 rounded-lg border border-surface-700 text-sm animate-fade-in">
          <div className="flex items-center justify-between mb-2">
            <span className="font-medium text-surface-200">
              {visibleSources[expandedSource].source}
            </span>
            <span className="text-xs text-surface-500">
              Page {visibleSources[expandedSource].page}
            </span>
          </div>
          <p className="text-surface-400 text-xs leading-relaxed">
            {visibleSources[expandedSource].preview}
          </p>
          <div className="mt-2 flex items-center justify-between">
            <span className="text-xs text-surface-600">
              Relevance: {(visibleSources[expandedSource].score * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Individual source chip
 */
function SourceChip({ source, isExpanded, onToggle }) {
  // Color code based on score
  const getScoreColor = (score) => {
    if (score >= 0.8) return 'text-emerald-400';
    if (score >= 0.6) return 'text-amber-400';
    return 'text-surface-400';
  };

  return (
    <button
      onClick={onToggle}
      className={`source-chip ${isExpanded ? 'border-primary-500 bg-primary-500/10' : ''}`}
    >
      <FileText className="w-3 h-3" />
      <span className="truncate max-w-[150px]">{source.source}</span>
      <span className="text-surface-500">p.{source.page}</span>
      <span className={`font-mono text-xs ${getScoreColor(source.score)}`}>
        {(source.score * 100).toFixed(0)}%
      </span>
    </button>
  );
}

export default SourcesList;
