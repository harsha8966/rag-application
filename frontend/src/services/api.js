/**
 * API Service - HTTP Client for Backend Communication
 * 
 * Centralizes all API calls with proper error handling.
 * Uses axios for HTTP requests with automatic JSON parsing.
 */

import axios from 'axios';

// API base URL - uses Vite proxy in development
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000, // 60 second timeout for LLM responses
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Extract error details
    const errorResponse = {
      message: 'An unexpected error occurred',
      details: null,
    };

    if (error.response) {
      // Server responded with error
      const data = error.response.data;
      errorResponse.message = data.message || data.detail?.message || 'Server error';
      errorResponse.details = data.details || data.detail?.details;
    } else if (error.request) {
      // Request made but no response
      errorResponse.message = 'Unable to connect to server';
    } else {
      // Error in request setup
      errorResponse.message = error.message;
    }

    return Promise.reject(errorResponse);
  }
);

/**
 * Upload a document for indexing
 * @param {File} file - The file to upload
 * @param {Function} onProgress - Progress callback (0-100)
 * @returns {Promise<Object>} Upload response
 */
export async function uploadDocument(file, onProgress = null) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 600000, // 10 minutes - large documents need parsing + embedding
    onUploadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const percentCompleted = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        );
        onProgress(percentCompleted);
      }
    },
  });

  return response.data;
}

/**
 * Get list of indexed documents
 * @returns {Promise<Object>} Document list
 */
export async function getDocuments() {
  const response = await api.get('/upload/documents');
  return response.data;
}

/**
 * Delete a document from the index
 * @param {string} documentId - Document ID to delete
 * @returns {Promise<Object>} Delete response
 */
export async function deleteDocument(documentId) {
  const response = await api.delete(`/upload/documents/${documentId}`);
  return response.data;
}

/**
 * Ask a question
 * @param {string} question - The question to ask
 * @param {Object} options - Additional options
 * @param {boolean} options.useMmr - Use MMR for diverse retrieval
 * @param {number} options.topK - Number of chunks to retrieve
 * @returns {Promise<Object>} Answer response
 */
export async function askQuestion(question, options = {}) {
  const response = await api.post('/ask', {
    question,
    use_mmr: options.useMmr ?? true,
    top_k: options.topK,
    session_id: options.sessionId,
  });

  return response.data;
}

/**
 * Submit feedback for a query
 * @param {string} queryId - Query ID from ask response
 * @param {string} feedbackType - 'positive', 'negative', 'partial', 'irrelevant'
 * @param {string} comment - Optional comment
 * @returns {Promise<Object>} Feedback response
 */
export async function submitFeedback(queryId, feedbackType, comment = null) {
  const response = await api.post('/feedback', {
    query_id: queryId,
    feedback_type: feedbackType,
    comment,
  });

  return response.data;
}

/**
 * Get feedback statistics
 * @returns {Promise<Object>} Feedback stats
 */
export async function getFeedbackStats() {
  const response = await api.get('/feedback/stats');
  return response.data;
}

/**
 * Health check
 * @returns {Promise<Object>} Health status
 */
export async function healthCheck() {
  const response = await api.get('/health');
  return response.data;
}

export default api;
