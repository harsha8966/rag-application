/**
 * File Upload Component
 * 
 * Handles document upload with:
 * - Drag and drop support
 * - File type validation
 * - Upload progress
 * - Document list display
 */

import React, { useState, useCallback } from 'react';
import { 
  Upload, 
  FileText, 
  CheckCircle, 
  XCircle, 
  Loader2,
  Trash2,
  MessageSquare,
  File,
  FileType
} from 'lucide-react';
import { deleteDocument } from '../services/api';

function FileUpload({ onUpload, documents, onViewChat }) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadState, setUploadState] = useState('idle'); // 'idle', 'uploading', 'success', 'error'
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadResult, setUploadResult] = useState(null);
  const [error, setError] = useState(null);

  // Allowed file types
  const ALLOWED_TYPES = ['.pdf', '.txt'];
  const ALLOWED_MIME = ['application/pdf', 'text/plain'];

  /**
   * Validate file before upload
   */
  const validateFile = (file) => {
    const extension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!ALLOWED_TYPES.includes(extension) && !ALLOWED_MIME.includes(file.type)) {
      return `Unsupported file type. Please upload ${ALLOWED_TYPES.join(' or ')} files.`;
    }
    
    // 50MB limit
    if (file.size > 50 * 1024 * 1024) {
      return 'File too large. Maximum size is 50MB.';
    }
    
    return null;
  };

  /**
   * Handle file upload
   */
  const handleUpload = async (file) => {
    const validationError = validateFile(file);
    if (validationError) {
      setError(validationError);
      setUploadState('error');
      return;
    }

    setUploadState('uploading');
    setUploadProgress(0);
    setError(null);

    try {
      const result = await onUpload(file, (progress) => {
        setUploadProgress(progress);
      });
      
      setUploadResult(result);
      setUploadState('success');
      
      // Reset after delay
      setTimeout(() => {
        setUploadState('idle');
        setUploadResult(null);
      }, 5000);
    } catch (err) {
      setError(err.message || 'Upload failed');
      setUploadState('error');
    }
  };

  /**
   * Handle drag events
   */
  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    if (file) {
      handleUpload(file);
    }
  }, []);

  /**
   * Handle file input change
   */
  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      handleUpload(file);
    }
    // Reset input
    e.target.value = '';
  };

  return (
    <div className="max-w-3xl mx-auto space-y-8">
      {/* Upload area */}
      <div className="card p-6">
        <h2 className="text-lg font-semibold text-surface-100 mb-4">
          Upload Documents
        </h2>
        
        <label
          className={`drop-zone flex flex-col items-center justify-center text-center ${
            isDragging ? 'drag-over' : ''
          } ${uploadState === 'uploading' ? 'pointer-events-none opacity-50' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <input
            type="file"
            accept=".pdf,.txt"
            onChange={handleFileSelect}
            className="hidden"
            disabled={uploadState === 'uploading'}
          />
          
          {uploadState === 'idle' && (
            <>
              <div className="w-14 h-14 rounded-xl bg-surface-800 border border-surface-700 flex items-center justify-center mb-4">
                <Upload className="w-7 h-7 text-surface-500" />
              </div>
              <p className="text-surface-300 mb-2">
                Drag & drop a file here, or click to browse
              </p>
              <p className="text-sm text-surface-600">
                Supports PDF and TXT files up to 50MB
              </p>
            </>
          )}

          {uploadState === 'uploading' && (
            <>
              <Loader2 className="w-10 h-10 text-primary-500 animate-spin mb-4" />
              <p className="text-surface-300 mb-2">Uploading and processing...</p>
              <div className="w-full max-w-xs h-2 bg-surface-800 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-primary-500 transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
              <p className="text-sm text-surface-600 mt-2">{uploadProgress}%</p>
            </>
          )}

          {uploadState === 'success' && uploadResult && (
            <>
              <CheckCircle className="w-10 h-10 text-emerald-500 mb-4" />
              <p className="text-emerald-400 mb-2">Upload successful!</p>
              <div className="text-sm text-surface-400 space-y-1">
                <p>{uploadResult.document.filename}</p>
                <p>{uploadResult.document.total_chunks} chunks created</p>
                <p>Processed in {uploadResult.processing_time_ms.toFixed(0)}ms</p>
              </div>
            </>
          )}

          {uploadState === 'error' && (
            <>
              <XCircle className="w-10 h-10 text-rose-500 mb-4" />
              <p className="text-rose-400 mb-2">Upload failed</p>
              <p className="text-sm text-surface-500">{error}</p>
              <button
                onClick={() => setUploadState('idle')}
                className="mt-4 text-sm text-primary-400 hover:text-primary-300"
              >
                Try again
              </button>
            </>
          )}
        </label>
      </div>

      {/* Document list */}
      <div className="card p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-surface-100">
            Indexed Documents
          </h2>
          <span className="text-sm text-surface-500">
            {documents.length} {documents.length === 1 ? 'document' : 'documents'}
          </span>
        </div>

        {documents.length === 0 ? (
          <div className="text-center py-8">
            <File className="w-12 h-12 text-surface-700 mx-auto mb-3" />
            <p className="text-surface-500">No documents uploaded yet</p>
            <p className="text-sm text-surface-600 mt-1">
              Upload files above to start asking questions
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            {documents.map((doc) => (
              <DocumentItem key={doc} document={doc} />
            ))}
          </div>
        )}

        {/* Go to chat button */}
        {documents.length > 0 && (
          <button
            onClick={onViewChat}
            className="btn-primary w-full mt-6 flex items-center justify-center gap-2"
          >
            <MessageSquare className="w-4 h-4" />
            Start Asking Questions
          </button>
        )}
      </div>
    </div>
  );
}

/**
 * Individual document item
 */
function DocumentItem({ document }) {
  const [isDeleting, setIsDeleting] = useState(false);

  const getFileIcon = () => {
    if (document.endsWith('.pdf')) {
      return <FileType className="w-5 h-5 text-rose-400" />;
    }
    return <FileText className="w-5 h-5 text-primary-400" />;
  };

  const handleDelete = async () => {
    if (!confirm(`Delete "${document}"? This cannot be undone.`)) return;
    
    setIsDeleting(true);
    try {
      await deleteDocument(document);
      // Parent will refresh the list
      window.location.reload();
    } catch (err) {
      alert('Failed to delete document');
      setIsDeleting(false);
    }
  };

  return (
    <div className="flex items-center justify-between p-3 bg-surface-800/50 rounded-lg border border-surface-700 group">
      <div className="flex items-center gap-3">
        {getFileIcon()}
        <span className="text-surface-200 truncate max-w-[300px]">
          {document}
        </span>
      </div>
      
      <button
        onClick={handleDelete}
        disabled={isDeleting}
        className="btn-icon opacity-0 group-hover:opacity-100 transition-opacity"
        title="Delete document"
      >
        {isDeleting ? (
          <Loader2 className="w-4 h-4 animate-spin" />
        ) : (
          <Trash2 className="w-4 h-4 text-surface-500 hover:text-rose-400" />
        )}
      </button>
    </div>
  );
}

export default FileUpload;
