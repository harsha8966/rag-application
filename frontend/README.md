#  RAG Frontend

React frontend for the  RAG Assistant.

## Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## Features

- **Chat Interface**: Clean, modern chat UI
- **File Upload**: Drag-and-drop document upload
- **Confidence Bar**: Visual confidence indicator
- **Source Citations**: Expandable source references
- **Feedback Buttons**: Thumbs up/down rating

## Tech Stack

- **React 18**: UI framework
- **Vite**: Build tool
- **Tailwind CSS**: Styling
- **Axios**: HTTP client
- **Lucide React**: Icons

## Project Structure

```
src/
├── components/
│   ├── ChatInterface.jsx    # Main chat UI
│   ├── MessageBubble.jsx    # Individual messages
│   ├── ConfidenceBar.jsx    # Confidence indicator
│   ├── SourcesList.jsx      # Source citations
│   ├── FeedbackButtons.jsx  # Rating buttons
│   └── FileUpload.jsx       # Document upload
│
├── services/
│   └── api.js               # Backend API client
│
├── styles/
│   └── main.css             # Tailwind + custom styles
│
├── App.jsx                  # Root component
└── index.jsx               # Entry point
```

## API Integration

The frontend communicates with the backend via:

```javascript
// Upload document
await uploadDocument(file);

// Ask question
const response = await askQuestion("What is the refund policy?");

// Submit feedback
await submitFeedback(queryId, "positive");
```

## Development

```bash
# Run with hot reload
npm run dev

# Lint code
npm run lint

# Preview production build
npm run preview
```

## Configuration

Create `.env.local` for local configuration:

```env
VITE_API_URL=http://localhost:8000
```

In development, Vite proxies `/api` requests to the backend.
