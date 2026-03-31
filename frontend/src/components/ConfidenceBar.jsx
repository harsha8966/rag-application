/**
 * Confidence Bar Component
 * 
 * Visual indicator showing how confident the system is in its answer.
 * Uses color coding:
 * - Green (high): 70%+ confidence
 * - Yellow (medium): 50-70% confidence  
 * - Red (low): Below 50% confidence
 */

import React from 'react';
import { Shield, ShieldAlert, ShieldCheck, Info } from 'lucide-react';

function ConfidenceBar({ confidence }) {
  const percentage = confidence.percentage;
  const level = confidence.level;
  
  // Determine colors and icon based on confidence level
  const getStyles = () => {
    switch (level) {
      case 'high':
        return {
          bgClass: 'bg-emerald-500/20',
          barClass: 'bg-emerald-500',
          textClass: 'text-emerald-400',
          Icon: ShieldCheck,
          label: 'High Confidence',
        };
      case 'medium':
        return {
          bgClass: 'bg-amber-500/20',
          barClass: 'bg-amber-500',
          textClass: 'text-amber-400',
          Icon: Shield,
          label: 'Medium Confidence',
        };
      case 'low':
      default:
        return {
          bgClass: 'bg-rose-500/20',
          barClass: 'bg-rose-500',
          textClass: 'text-rose-400',
          Icon: ShieldAlert,
          label: 'Low Confidence',
        };
    }
  };

  const styles = getStyles();
  const { Icon } = styles;

  return (
    <div className="space-y-2">
      {/* Label and percentage */}
      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center gap-2">
          <Icon className={`w-4 h-4 ${styles.textClass}`} />
          <span className={styles.textClass}>{styles.label}</span>
        </div>
        <span className="text-surface-400 font-mono">{percentage}%</span>
      </div>

      {/* Progress bar */}
      <div className={`h-2 rounded-full ${styles.bgClass} overflow-hidden`}>
        <div
          className={`h-full ${styles.barClass} rounded-full transition-all duration-500`}
          style={{ width: `${percentage}%` }}
        />
      </div>

      {/* Explanation */}
      {confidence.explanation && (
        <div className="flex items-start gap-2 text-xs text-surface-500">
          <Info className="w-3 h-3 mt-0.5 flex-shrink-0" />
          <span>{confidence.explanation}</span>
        </div>
      )}
    </div>
  );
}

export default ConfidenceBar;
