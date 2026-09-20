# AI-Powered Resume Analyzer and Job Matching System

An intelligent web application that uses AI and NLP to analyze resumes, match them against job descriptions, and provide actionable improvement recommendations.

## Features

- **Resume Upload**: Support for PDF and DOCX formats
- **AI Analysis**: Automatic extraction of skills, education, experience, and certifications
- **Job Matching**: Multi-dimensional match scoring (Skills, Experience, Education, Semantic Similarity)
- **Smart Recommendations**: Prioritized, actionable suggestions for resume improvement
- **ATS Optimization**: Tips to optimize resumes for Applicant Tracking Systems
- **Analysis History**: Track and compare past analyses

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18 + TypeScript + Vite |
| Backend | Python 3.11+ + FastAPI |
| Database | MongoDB 7 |
| AI/NLP | spaCy, Sentence Transformers, Scikit-learn |
| File Processing | pdfplumber, PyPDF2, python-docx |

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker Desktop (for MongoDB)

### 1. Start MongoDB
```bash
docker-compose up -d
```

### 2. Setup Backend
```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Start server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`
- Swagger Docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 3. Setup Frontend
```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

The app will be available at `http://localhost:5173`

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/analyze` | Upload resume + JD for analysis |
| `GET` | `/api/v1/analyses` | List past analyses |
| `GET` | `/api/v1/analyses/{id}` | Get analysis details |
| `DELETE` | `/api/v1/analyses/{id}` | Delete an analysis |
| `GET` | `/api/v1/health` | Health check |

## Scoring Algorithm

The overall score is a weighted combination of four components:

| Component | Weight | Method |
|-----------|--------|--------|
| Skills Match | 40% | Jaccard similarity against skills taxonomy |
| Experience | 25% | Years comparison with role relevance |
| Education | 15% | Degree level hierarchy comparison |
| Semantic Similarity | 20% | Sentence Transformer cosine similarity |

## Project Structure

```
resume-analyzer/
├── docker-compose.yml          # MongoDB container
├── backend/
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # Settings
│   ├── api/routes/             # API endpoints
│   ├── api/schemas/            # Pydantic models
│   ├── core/                   # Database & file processing
│   ├── nlp/                    # AI/NLP pipeline
│   │   ├── extractor.py        # Information extraction
│   │   ├── matcher.py          # Matching algorithms
│   │   ├── scorer.py           # Score computation
│   │   └── recommender.py      # Recommendations engine
│   └── data/                   # Skills taxonomy & ATS data
└── frontend/
    ├── src/
    │   ├── App.tsx             # Main application
    │   ├── index.css           # Design system
    │   ├── services/api.ts     # API client
    │   └── types/index.ts      # TypeScript types
    └── index.html
```

## License

MIT
