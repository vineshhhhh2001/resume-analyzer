# Release Notes

## v1.0.0 - Initial Release

### Date: 2026-09-20

### New Features
- Resume upload support for PDF and DOCX formats
- AI-powered resume analysis using spaCy and Sentence Transformers
- Job description matching with multi-dimensional scoring
- Smart recommendations engine for resume improvement
- ATS optimization tips
- Analysis history tracking
- React + TypeScript frontend with modern UI

### Technical Stack
- **Frontend**: React 18, TypeScript, Vite
- **Backend**: Python 3.11+, FastAPI
- **Database**: MongoDB 7
- **AI/NLP**: spaCy, Sentence Transformers, Scikit-learn, OpenAI, Google Gemini, Groq
- **File Processing**: pdfplumber, PyPDF2, python-docx
- **Containerization**: Docker, Docker Compose

### Scoring Algorithm
| Component | Weight |
|-----------|--------|
| Skills Match | 40% |
| Experience | 25% |
| Education | 15% |
| Semantic Similarity | 20% |

### Known Issues
- None at initial release

### Contributors
- Vinesh
