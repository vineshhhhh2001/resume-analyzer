import { useState, useEffect, useCallback, useRef } from 'react';
import type {
  AnalysisResult,
  AnalysisListItem,
  CategoryScores,
  Recommendation,
  ATSSuggestion,
} from './types';
import { analyzeResume, getAnalyses, getAnalysis, deleteAnalysis } from './services/api';

// ─── Hash-based Router ────────────────────────────────────────────────────

type Page = 'home' | 'analyze' | 'results';

function useRouter() {
  const [page, setPage] = useState<Page>('home');
  const [resultId, setResultId] = useState<string | null>(null);

  useEffect(() => {
    function handleHash() {
      const hash = window.location.hash.slice(1);
      if (hash.startsWith('results/')) {
        setPage('results');
        setResultId(hash.split('/')[1]);
      } else if (hash === 'analyze') {
        setPage('analyze');
        setResultId(null);
      } else {
        setPage('home');
        setResultId(null);
      }
    }

    handleHash();
    window.addEventListener('hashchange', handleHash);
    return () => window.removeEventListener('hashchange', handleHash);
  }, []);

  const navigate = useCallback((target: string) => {
    window.location.hash = target;
  }, []);

  return { page, resultId, navigate };
}

// ─── Score Circle Component ───────────────────────────────────────────────

function ScoreCircle({
  score,
  grade,
  color,
  size = 200,
}: {
  score: number;
  grade: string;
  color: string;
  size?: number;
}) {
  const [animatedScore, setAnimatedScore] = useState(0);
  const radius = (size - 16) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (animatedScore / 100) * circumference;

  useEffect(() => {
    const timer = setTimeout(() => setAnimatedScore(score), 100);
    return () => clearTimeout(timer);
  }, [score]);

  // Animate the number
  const [displayScore, setDisplayScore] = useState(0);
  useEffect(() => {
    let start = 0;
    const end = score;
    const duration = 1500;
    const startTime = performance.now();

    function animate(now: number) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // easeOutCubic
      setDisplayScore(Math.round(start + (end - start) * eased));
      if (progress < 1) requestAnimationFrame(animate);
    }

    requestAnimationFrame(animate);
  }, [score]);

  return (
    <div className="score-circle" style={{ width: size, height: size }}>
      <svg width={size} height={size}>
        <circle
          className="score-circle-bg"
          cx={size / 2}
          cy={size / 2}
          r={radius}
        />
        <circle
          className="score-circle-progress"
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={color}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{ filter: `drop-shadow(0 0 8px ${color}40)` }}
        />
      </svg>
      <div className="score-circle-text">
        <div className="score-number" style={{ color }}>
          {displayScore}
        </div>
        <div className="score-label">out of 100</div>
        <div className="score-grade" style={{ color }}>
          {grade}
        </div>
      </div>
    </div>
  );
}

// ─── Progress Bar Component ───────────────────────────────────────────────

function ProgressBar({
  value,
  label,
  showValue = true,
}: {
  value: number;
  label: string;
  showValue?: boolean;
}) {
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => setWidth(value), 200);
    return () => clearTimeout(timer);
  }, [value]);

  const colorClass = value >= 70 ? 'success' : value >= 45 ? 'warning' : 'danger';
  const colorValue =
    value >= 70 ? '#10b981' : value >= 45 ? '#f59e0b' : '#ef4444';

  return (
    <div className="category-score-item card">
      <div className="category-score-header">
        <span className="category-score-label">{label}</span>
        {showValue && (
          <span className="category-score-value" style={{ color: colorValue }}>
            {Math.round(value)}%
          </span>
        )}
      </div>
      <div className="progress-bar">
        <div
          className={`progress-bar-fill ${colorClass}`}
          style={{ width: `${width}%` }}
        />
      </div>
    </div>
  );
}

// ─── Loading Overlay ──────────────────────────────────────────────────────

function LoadingOverlay({ step }: { step: number }) {
  const steps = [
    'Uploading candidate credentials...',
    'Extracting resume text content...',
    'Analyzing skills, education, & experience...',
    'Computing recruitment compatibility scores...',
    'Generating hiring assessment matrix...',
  ];

  return (
    <div className="loading-overlay">
      <div className="loading-spinner" />
      <div className="loading-text">Auditing Candidate Credentials</div>
      <div className="loading-subtext" style={{ marginTop: 8, color: 'var(--text-secondary)' }}>
        Current Step: {steps[step] || 'Processing...'}
      </div>
    </div>
  );
}

// ─── Header Component ─────────────────────────────────────────────────────

function Header({
  page,
  navigate,
}: {
  page: Page;
  navigate: (target: string) => void;
}) {
  return (
    <header className="header">
      <div className="header-inner">
        <div className="header-logo" onClick={() => navigate('')}>
          <span className="header-logo-icon">💼</span>
          <span className="header-logo-gradient">TalentFlow AI</span>
          <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-tertiary)', fontWeight: 500, marginLeft: 8, borderLeft: '1px solid var(--glass-border)', paddingLeft: 8 }}>Recruiter Portal</span>
        </div>
      </div>
    </header>
  );
}

// ─── Home Page ────────────────────────────────────────────────────────────

function HomePage({
  navigate,
}: {
  navigate: (target: string) => void;
}) {
  const [analyses, setAnalyses] = useState<AnalysisListItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [showHistory, setShowHistory] = useState(true); // default to true for HR tracking
  const [searchTerm, setSearchTerm] = useState('');
  const [scoreFilter, setScoreFilter] = useState('all');
  const [sortBy, setSortBy] = useState('date-desc');

  const filteredAnalyses = analyses
    .filter((a) => {
      const matchText = a.resume_filename.toLowerCase().includes(searchTerm.toLowerCase());
      if (!matchText) return false;

      if (scoreFilter === 'excellent') {
        return a.overall_score >= 75;
      } else if (scoreFilter === 'good') {
        return a.overall_score >= 50 && a.overall_score < 75;
      } else if (scoreFilter === 'poor') {
        return a.overall_score < 50;
      }
      return true;
    })
    .sort((a, b) => {
      if (sortBy === 'date-desc') {
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      } else if (sortBy === 'date-asc') {
        return new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
      } else if (sortBy === 'score-desc') {
        return b.overall_score - a.overall_score;
      } else if (sortBy === 'score-asc') {
        return a.overall_score - b.overall_score;
      }
      return 0;
    });

  useEffect(() => {
    loadAnalyses();
  }, []);

  async function loadAnalyses() {
    try {
      setLoading(true);
      const data = await getAnalyses();
      setAnalyses(data.analyses);
      setTotal(data.total);
    } catch {
      // API offline fallback
    } finally {
      setLoading(false);
    }
  }

  async function handleDelete(id: string) {
    if (!confirm('Delete this screening dossier?')) return;
    try {
      await deleteAnalysis(id);
      loadAnalyses();
    } catch {
      alert('Failed to delete screening dossier.');
    }
  }

  const shortlistedCount = analyses.filter((a) => a.overall_score >= 75).length;
  const phoneScreenCount = analyses.filter((a) => a.overall_score >= 50 && a.overall_score < 75).length;
  const avgScore =
    analyses.length > 0
      ? Math.round(
          analyses.reduce((sum, a) => sum + a.overall_score, 0) / analyses.length,
        )
      : 0;

  const shortlistRate = analyses.length > 0 ? Math.round((shortlistedCount / analyses.length) * 100) : 0;

  return (
    <>
      {/* Hero */}
      <section className="hero">
        <div className="hero-content">
          <div className="hero-badge animate-in">⚡ Automated Candidate Qualification</div>
          <h1 className="animate-in animate-delay-1">
            Screen and Audit Candidates with{' '}
            <span className="gradient-text">AI Precision</span>
          </h1>
          <p className="animate-in animate-delay-2">
            Automate your recruiting workflow. Upload resumes against job descriptions to
            verify skills matching, calculate experience compliance, perform credential checks, and generate recruiter screening questions.
          </p>
          <button
            className="btn btn-primary btn-lg animate-in animate-delay-3"
            onClick={() => navigate('analyze')}
          >
            ⚡ Screen New Candidate
          </button>
        </div>
      </section>

      <div className="page">
        {/* HR Dashboard Features */}
        <div className="features-grid animate-in animate-delay-4">
          {[
            {
              icon: '⚖️',
              title: 'Hiring Decision Verdicts',
              desc: 'Instantly categorizes candidates into Auto-Shortlist, Recruiter Phone Screen, or Mismatch Hold pipelines.',
            },
            {
              icon: '📋',
              title: 'Compliance Check Matrix',
              desc: 'Automates verification checks on minimum years of experience, core technical skills, and degree matching.',
            },
            {
              icon: '💡',
              title: 'Hiring Insights & Gaps',
              desc: 'Identifies education/experience gaps, candidate strengths, and advice to support candidate engagement.',
            },
            {
              icon: '🎯',
              title: 'Keyword Skill Mapping',
              desc: 'Identifies matching required technologies, extra strengths, and missing prerequisites at a glance.',
            },
          ].map((feature) => (
            <div key={feature.title} className="feature-card card card-glow">
              <div className="feature-icon">{feature.icon}</div>
              <h3>{feature.title}</h3>
              <p>{feature.desc}</p>
            </div>
          ))}
        </div>

        {/* Recruiter Dashboard Stats */}
        {analyses.length > 0 && (
          <div style={{ marginTop: 'var(--space-2xl)' }}>
            <h2 className="section-title">📊 Recruiter Analytics Overview</h2>
            <div className="stats-grid">
              <div className="stat-card card">
                <div className="stat-value">{total}</div>
                <div className="stat-label">Total Audited</div>
              </div>
              <div className="stat-card card">
                <div className="stat-value" style={{ color: 'var(--color-success)' }}>{shortlistRate}%</div>
                <div className="stat-label">Auto-Shortlist Rate</div>
              </div>
              <div className="stat-card card">
                <div className="stat-value" style={{ color: 'var(--color-warning)' }}>{phoneScreenCount}</div>
                <div className="stat-label">Pending Phone Screens</div>
              </div>
              <div className="stat-card card">
                <div className="stat-value" style={{ color: 'var(--accent-primary-light)' }}>{avgScore}%</div>
                <div className="stat-label">Average Match Score</div>
              </div>
            </div>
          </div>
        )}

        {/* Screening Pipeline */}
        <div style={{ marginTop: 'var(--space-2xl)' }}>
          {loading ? (
            <div className="empty-state">
              <div className="loading-spinner" style={{ margin: '0 auto' }} />
              <p style={{ marginTop: 16 }}>Loading recruiting pipeline...</p>
            </div>
          ) : analyses.length === 0 ? (
            <>
              <h2 className="section-title">📁 Screening Pipeline</h2>
              <div className="empty-state">
                <div className="empty-state-icon">📁</div>
                <h3>No candidates screened yet</h3>
                <p>Begin auditing candidates by uploading a resume and configuring job requirements.</p>
                <button
                  className="btn btn-primary"
                  style={{ marginTop: 16 }}
                  onClick={() => navigate('analyze')}
                >
                  ⚡ Screen First Candidate
                </button>
              </div>
            </>
          ) : (
            <>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-md)' }}>
                <h2 className="section-title" style={{ margin: 0 }}>📁 Candidate Screening Pipeline</h2>
                <button
                  className="btn btn-secondary"
                  onClick={() => setShowHistory((prev) => !prev)}
                  style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}
                >
                  {showHistory ? 'Hide Pipeline ▴' : 'Show Pipeline ▾'}
                </button>
              </div>

              {showHistory && (
                <div className="animate-in" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
                  {/* Filter Bar */}
                  <div className="filter-bar">
                    <div className="filter-group">
                      <label htmlFor="search-input" className="filter-label">Search Candidate/File</label>
                      <input
                        id="search-input"
                        type="text"
                        className="filter-input"
                        placeholder="Search candidate name or file..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                      />
                    </div>
                    <div className="filter-group">
                      <label htmlFor="score-filter" className="filter-label">Recruiter Verdict</label>
                      <select
                        id="score-filter"
                        className="filter-select"
                        value={scoreFilter}
                        onChange={(e) => setScoreFilter(e.target.value)}
                      >
                        <option value="all">All Verdicts</option>
                        <option value="excellent">Auto-Shortlist (75%+)</option>
                        <option value="good">Phone Screen (50% - 74%)</option>
                        <option value="poor">Mismatch Hold ({"< 50%"})</option>
                      </select>
                    </div>
                    <div className="filter-group">
                      <label htmlFor="sort-by" className="filter-label">Sort By</label>
                      <select
                        id="sort-by"
                        className="filter-select"
                        value={sortBy}
                        onChange={(e) => setSortBy(e.target.value)}
                      >
                        <option value="date-desc">Audit Date (Newest First)</option>
                        <option value="date-asc">Audit Date (Oldest First)</option>
                        <option value="score-desc">Match Score (Highest First)</option>
                        <option value="score-asc">Match Score (Lowest First)</option>
                      </select>
                    </div>
                  </div>

                  {filteredAnalyses.length === 0 ? (
                    <div className="empty-state card" style={{ padding: 'var(--space-xl)' }}>
                      <div className="empty-state-icon" style={{ fontSize: '2rem', marginBottom: 'var(--space-sm)' }}>🔍</div>
                      <h3 style={{ color: 'var(--text-secondary)' }}>No matching dossiers found</h3>
                      <p style={{ fontSize: 'var(--font-size-sm)' }}>Adjust your search query or filters to find candidate records.</p>
                    </div>
                  ) : (
                    <div className="history-table-wrapper">
                      <table className="history-table">
                        <thead>
                          <tr>
                            <th>Candidate Resume</th>
                            <th>Target Role Position</th>
                            <th>AI Score</th>
                            <th>Recruitment Verdict</th>
                            <th>Audit Date</th>
                            <th>Actions</th>
                          </tr>
                        </thead>
                        <tbody>
                          {filteredAnalyses.map((a) => {
                            const score = a.overall_score;
                            const verdictLabel = score >= 75 ? 'Auto-Shortlist' : score >= 50 ? 'Phone Screen' : 'Mismatch Hold';
                            const verdictColorClass = score >= 75 ? 'badge-success' : score >= 50 ? 'badge-warning' : 'badge-danger';
                            const verdictColorValue = score >= 75 ? '#10b981' : score >= 50 ? '#f59e0b' : '#ef4444';

                            return (
                              <tr key={a.id}>
                                <td style={{ color: 'var(--text-primary)', fontWeight: 500 }}>
                                  📄 {a.resume_filename}
                                </td>
                                <td>
                                  {a.job_title || 'General Screening'}
                                </td>
                                <td>
                                  <span
                                    className="history-score"
                                    style={{ color: a.grade.color, fontWeight: 700 }}
                                  >
                                    {Math.round(score)}%
                                  </span>
                                </td>
                                <td>
                                  <span className={`badge ${verdictColorClass}`}>
                                    {verdictLabel}
                                  </span>
                                </td>
                                <td style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)' }}>
                                  {new Date(a.created_at).toLocaleDateString('en-US', {
                                    month: 'short',
                                    day: 'numeric',
                                    year: 'numeric',
                                    hour: '2-digit',
                                    minute: '2-digit',
                                  })}
                                </td>
                                <td>
                                  <div className="history-actions">
                                    <button
                                      className="btn btn-primary"
                                      onClick={() => navigate(`results/${a.id}`)}
                                      style={{ padding: '4px 12px', fontSize: 'var(--font-size-xs)', borderRadius: 'var(--radius-sm)' }}
                                    >
                                      Open Dossier
                                    </button>
                                    <button
                                      className="btn btn-danger"
                                      onClick={() => handleDelete(a.id)}
                                      style={{ padding: '4px 8px', fontSize: 'var(--font-size-xs)', borderRadius: 'var(--radius-sm)' }}
                                    >
                                      ✕
                                    </button>
                                  </div>
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </>
  );
}

// ─── Analyze Page ─────────────────────────────────────────────────────────

function AnalyzePage({ navigate }: { navigate: (target: string) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [jobDescription, setJobDescription] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(true);
  }

  function handleDragLeave(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) validateAndSetFile(dropped);
  }

  function handleFileSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const selected = e.target.files?.[0];
    if (selected) validateAndSetFile(selected);
  }

  function validateAndSetFile(f: File) {
    const ext = f.name.toLowerCase().split('.').pop();
    if (ext !== 'pdf' && ext !== 'docx' && ext !== 'doc') {
      setError('Please upload a PDF or DOCX file.');
      return;
    }
    if (f.size > 10 * 1024 * 1024) {
      setError('File size must be under 10MB.');
      return;
    }
    setFile(f);
    setError(null);
  }

  async function handleAnalyze() {
    if (!file || !jobDescription.trim()) return;

    setLoading(true);
    setError(null);
    setLoadingStep(0);

    const stepTimer = setInterval(() => {
      setLoadingStep((prev) => Math.min(prev + 1, 4));
    }, 2500);

    try {
      const result = await analyzeResume(file, jobDescription);
      clearInterval(stepTimer);
      navigate(`results/${result.id}`);
    } catch (err: any) {
      clearInterval(stepTimer);
      const message =
        err.response?.data?.detail ||
        err.message ||
        'Analysis failed. Make sure the backend server is running.';
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  const wordCount = jobDescription
    .trim()
    .split(/\s+/)
    .filter((w) => w).length;
  const canAnalyze = file && jobDescription.trim().length > 20 && !loading;

  return (
    <>
      {loading && <LoadingOverlay step={loadingStep} />}

      <div className="page">
        <div className="page-header">
          <h1 className="page-title">📄 Audit Candidate Qualification</h1>
          <p className="page-subtitle">
            Upload candidate credentials and specify job parameters to run an automated hiring suitability check.
          </p>
        </div>

        {error && (
          <div
            style={{
              padding: '12px 16px',
              background: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid rgba(239, 68, 68, 0.2)',
              borderRadius: 'var(--radius-md)',
              color: 'var(--color-danger-light)',
              fontSize: 'var(--font-size-sm)',
              marginBottom: 'var(--space-lg)',
            }}
          >
            ⚠️ {error}
          </div>
        )}

        <div className="analyze-layout">
          {/* Resume Upload */}
          <div>
            <h3 className="section-title">📎 Candidate Resume</h3>
            <div
              className={`upload-zone ${dragOver ? 'drag-over' : ''} ${file ? 'has-file' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.docx,.doc"
                onChange={handleFileSelect}
                style={{ display: 'none' }}
                id="resume-upload-input"
              />
              {file ? (
                <>
                  <div className="upload-icon">✅</div>
                  <div className="upload-text">
                    Resume file loaded successfully
                  </div>
                  <div className="upload-file-info">
                    <span>📄</span>
                    <span className="upload-file-name">{file.name}</span>
                    <span style={{ color: 'var(--text-tertiary)' }}>
                      ({(file.size / 1024).toFixed(0)} KB)
                    </span>
                    <button
                      className="upload-file-remove"
                      onClick={(e) => {
                        e.stopPropagation();
                        setFile(null);
                      }}
                      title="Remove file"
                    >
                      ✕
                    </button>
                  </div>
                </>
              ) : (
                <>
                  <div className="upload-icon">📤</div>
                  <div className="upload-text">
                    <strong>Click to upload resume</strong> or drag and drop
                  </div>
                  <div className="upload-accepted">
                    Supports PDF, DOCX • Max 10MB
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Job Description */}
          <div>
            <h3 className="section-title">📋 Target Job Description</h3>
            <div className="jd-input-wrapper">
              <textarea
                className="jd-textarea"
                placeholder={`Paste the job description or role requirements here...\n\nExample:\nSenior React Developer\n- 4+ years of JavaScript/TypeScript experience\n- Advanced knowledge of React and frontend architecture\n- Experience with Node.js and REST APIs\n- BS in CS or equivalent experience`}
                value={jobDescription}
                onChange={(e) => setJobDescription(e.target.value)}
                id="job-description-input"
              />
              <div className="jd-word-count">{wordCount} words</div>
            </div>
          </div>
        </div>

        {/* Analyze Button */}
        <div style={{ textAlign: 'center', marginTop: 'var(--space-lg)' }}>
          <button
            className="btn btn-primary btn-lg"
            onClick={handleAnalyze}
            disabled={!canAnalyze}
            id="analyze-button"
            style={{ minWidth: 260 }}
          >
            {loading ? (
              <>
                <span className="loading-spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />{' '}
                Auditing candidate...
              </>
            ) : (
              <>⚡ Run Candidate Audit</>
            )}
          </button>
          {!canAnalyze && !loading && (
            <p
              style={{
                color: 'var(--text-tertiary)',
                fontSize: 'var(--font-size-xs)',
                marginTop: 8,
              }}
            >
              {!file
                ? 'Upload a candidate resume to continue'
                : 'Enter a job description (min 20 characters)'}
            </p>
          )}
        </div>
      </div>
    </>
  );
}

// ─── Results Page ─────────────────────────────────────────────────────────

function ResultsPage({
  resultId,
  navigate,
}: {
  resultId: string;
  navigate: (target: string) => void;
}) {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'summary' | 'skills' | 'profile' | 'feedback'>('summary');

  useEffect(() => {
    loadResult();
  }, [resultId]);

  async function loadResult() {
    try {
      setLoading(true);
      const data = await getAnalysis(resultId);
      setResult(data);
    } catch {
      setError('Failed to load candidate screening dossier.');
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return (
      <div className="page">
        <div className="empty-state">
          <div className="loading-spinner" style={{ margin: '0 auto' }} />
          <p style={{ marginTop: 16 }}>Loading qualification dossier...</p>
        </div>
      </div>
    );
  }

  if (error || !result) {
    return (
      <div className="page">
        <div className="empty-state">
          <div className="empty-state-icon">⚠️</div>
          <h3>{error || 'Dossier not found'}</h3>
          <button
            className="btn btn-primary"
            style={{ marginTop: 16 }}
            onClick={() => navigate('')}
          >
            Back to Pipeline
          </button>
        </div>
      </div>
    );
  }

  // HR Verdict styling
  const score = result.overall_score;
  const assessment = result.hiring_assessment || {
    verdict: score >= 75 ? "Auto-Shortlist (Strong Match)" : score >= 50 ? "Manual Review / Phone Screen" : "Hold / Mismatch",
    verdict_color: score >= 75 ? "#10b981" : score >= 50 ? "#f59e0b" : "#ef4444",
    leveling: result.resume_data.total_experience_years >= 6 ? "Senior Level" : result.resume_data.total_experience_years >= 3 ? "Mid-Level" : "Junior Level",
    summary: `Candidate meets ${score.toFixed(0)}% of requirements.`,
    checklist: [],
    screening_actions: []
  };

  const categoryLabels: Record<string, string> = {
    skills_match: '🎯 Core Skills Align',
    experience_match: '💼 Experience Scope',
    education_match: '🎓 Education Match',
    semantic_similarity: '📊 Resume Relevance',
  };

  return (
    <div className="page">
      {/* Back button */}
      <button
        className="btn btn-ghost"
        onClick={() => navigate('')}
        style={{ marginBottom: 'var(--space-lg)' }}
      >
        ← Back to Recruiting Pipeline
      </button>

      {/* Main Dossier Header Sheet */}
      <div className="report-sheet" style={{ marginBottom: 'var(--space-lg)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 'var(--space-lg)' }}>
          <div>
            <span style={{ fontSize: 'var(--font-size-xs)', textTransform: 'uppercase', color: 'var(--accent-primary-light)', fontWeight: 700, letterSpacing: 1 }}>
              Recruiter Audit Record
            </span>
            <h1 className="page-title" style={{ margin: '4px 0 8px 0' }}>
              {result.resume_data.name || result.resume_filename.split('.')[0]}
            </h1>
            <p className="page-subtitle" style={{ fontSize: 'var(--font-size-sm)' }}>
              📄 File: <span style={{ color: 'var(--text-primary)' }}>{result.resume_filename}</span> 
              {result.job_data.title && (
                <> • Position: <span style={{ color: 'var(--text-primary)' }}>{result.job_data.title}</span></>
              )}
            </p>
            <p style={{ color: 'var(--text-tertiary)', fontSize: 'var(--font-size-xs)', marginTop: 6 }}>
              Screened on {new Date(result.created_at).toLocaleDateString('en-US', {
                month: 'long',
                day: 'numeric',
                year: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
              })}
            </p>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 8 }}>
            <div className="dossier-verdict-badge" style={{ color: assessment.verdict_color }}>
              <span>⚖️</span> {assessment.verdict}
            </div>
            <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)', fontWeight: 600 }}>
              Audit Score: <span style={{ color: 'var(--text-primary)', fontSize: 'var(--font-size-sm)' }}>{Math.round(score)}/100</span>
            </div>
          </div>
        </div>
      </div>

      {/* Recruiter Tabs Navigation */}
      <div className="hr-tabs">
        <button
          className={`hr-tab-btn ${activeTab === 'summary' ? 'active' : ''}`}
          onClick={() => setActiveTab('summary')}
        >
          📋 Recruiter Summary & Verdict
        </button>
        <button
          className={`hr-tab-btn ${activeTab === 'skills' ? 'active' : ''}`}
          onClick={() => setActiveTab('skills')}
        >
          🎯 Technical Match Matrix
        </button>
        <button
          className={`hr-tab-btn ${activeTab === 'profile' ? 'active' : ''}`}
          onClick={() => setActiveTab('profile')}
        >
          👤 Candidate Profile Dossier
        </button>
      </div>

      {/* TAB CONTENT: SUMMARY & VERDICT */}
      {activeTab === 'summary' && (
        <div className="dossier-grid animate-in">
          {/* Left: Summary and Action items */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)' }}>
            
            {/* Executive Summary */}
            <div className="dossier-card">
              <h3 className="dossier-section-title"><span>📝</span> Executive Qualification Summary</h3>
              <p style={{ fontSize: 'var(--font-size-base)', lineHeight: 1.7, color: 'var(--text-secondary)', whiteSpace: 'pre-wrap' }}>
                {assessment.summary}
              </p>
              <div style={{ display: 'flex', gap: 16, marginTop: 16 }}>
                <div>
                  <strong style={{ display: 'block', fontSize: 'var(--font-size-xs)', color: 'var(--text-tertiary)', textTransform: 'uppercase' }}>Target Classification</strong>
                  <span style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-primary)' }}>💼 {assessment.leveling}</span>
                </div>
                <div>
                  <strong style={{ display: 'block', fontSize: 'var(--font-size-xs)', color: 'var(--text-tertiary)', textTransform: 'uppercase' }}>Overall Score</strong>
                  <span style={{ fontSize: 'var(--font-size-sm)', color: assessment.verdict_color, fontWeight: 'bold' }}>{Math.round(score)}% Alignment</span>
                </div>
              </div>
            </div>





          </div>

          {/* Right: Verification Matrix */}
          <div>
            <div className="dossier-card">
              <h3 className="dossier-section-title"><span>📋</span> Recruitment Compliance Checks</h3>
              {assessment.checklist && assessment.checklist.length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                  {assessment.checklist.map((item, idx) => (
                    <div key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', paddingBottom: 12 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                        <strong style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-primary)' }}>
                          {item.criteria}
                        </strong>
                        <span className={`badge ${item.is_met ? 'badge-success' : 'badge-danger'}`} style={{ fontSize: '10px' }}>
                          {item.status}
                        </span>
                      </div>
                      <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)' }}>
                        {item.details}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-tertiary)', textAlign: 'center', padding: '20px 0' }}>
                  No checklists computed for this run.
                </p>
              )}
            </div>
            
            <div className="dossier-card">
              <h3 className="dossier-section-title"><span>📈</span> Structural Scores</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {Object.entries(result.category_scores).map(([key, val]) => (
                  <div key={key}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 'var(--font-size-xs)', marginBottom: 4 }}>
                      <span style={{ color: 'var(--text-secondary)' }}>{categoryLabels[key] || key}</span>
                      <strong style={{ color: val >= 70 ? 'var(--color-success)' : val >= 50 ? 'var(--color-warning)' : 'var(--color-danger)' }}>{Math.round(val)}%</strong>
                    </div>
                    <div className="progress-bar">
                      <div
                        className={`progress-bar-fill ${val >= 70 ? 'success' : val >= 50 ? 'warning' : 'danger'}`}
                        style={{ width: `${val}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB CONTENT: SKILLS & MATRICES */}
      {activeTab === 'skills' && (
        <div className="dossier-card animate-in" style={{ padding: 'var(--space-xl)' }}>
          {/* Main score details */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 'var(--space-xl)', marginBottom: 'var(--space-2xl)', alignItems: 'center' }}>
            <div className="score-card-container">
              <ScoreCircle
                score={result.overall_score}
                grade={assessment.verdict.split(' (')[0]}
                color={assessment.verdict_color}
              />
            </div>
            <div>
              <h3 className="dossier-section-title"><span>📊</span> Technical Competence Alignment</h3>
              <p style={{ color: 'var(--text-secondary)', fontSize: 'var(--font-size-sm)', lineHeight: 1.6, marginBottom: 16 }}>
                The compatibility scores measure four dimensions. Technical skills alignment acts as the core baseline, experience years checks the duration depth, education evaluates level relevance, and semantic similarity tracks total language overlap.
              </p>
              
              <div className="category-scores">
                {Object.entries(result.category_scores).map(([key, value]) => (
                  <ProgressBar
                    key={key}
                    value={value}
                    label={categoryLabels[key] || key}
                  />
                ))}
              </div>
            </div>
          </div>

          {/* Skill Keywords board */}
          <h3 className="dossier-section-title"><span>🏷️</span> Skill Keywords Compliance Audit</h3>
          <div className="skills-grid" style={{ marginTop: 'var(--space-lg)' }}>
            
            {/* Matched */}
            <div className="dossier-card" style={{ borderTop: '3px solid var(--color-success)' }}>
              <h4 style={{ color: 'var(--color-success-light)', fontSize: 'var(--font-size-sm)' }}>
                ✓ Matched Core Skills ({result.skills_detail.matched_skills.length})
              </h4>
              {result.skills_detail.matched_skills.length > 0 ? (
                <div className="skills-tags">
                  {result.skills_detail.matched_skills.map((s) => (
                    <span key={s} className="badge badge-success">{s}</span>
                  ))}
                </div>
              ) : (
                <p style={{ color: 'var(--text-tertiary)', fontSize: 'var(--font-size-xs)' }}>Zero exact skill matches.</p>
              )}
            </div>

            {/* Missing Required */}
            <div className="dossier-card" style={{ borderTop: '3px solid var(--color-danger)' }}>
              <h4 style={{ color: 'var(--color-danger-light)', fontSize: 'var(--font-size-sm)' }}>
                ✗ Missing Required Skills ({result.skills_detail.missing_required.length})
              </h4>
              {result.skills_detail.missing_required.length > 0 ? (
                <div className="skills-tags">
                  {result.skills_detail.missing_required.map((s) => (
                    <span key={s} className="badge badge-danger">{s}</span>
                  ))}
                </div>
              ) : (
                <p style={{ color: 'var(--text-secondary)', fontSize: 'var(--font-size-xs)' }}>✓ No missing required skills!</p>
              )}
            </div>

            {/* Missing Preferred */}
            <div className="dossier-card" style={{ borderTop: '3px solid var(--color-warning)' }}>
              <h4 style={{ color: 'var(--color-warning-light)', fontSize: 'var(--font-size-sm)' }}>
                ◯ Missing Nice-to-Have Skills ({result.skills_detail.missing_preferred.length})
              </h4>
              {result.skills_detail.missing_preferred.length > 0 ? (
                <div className="skills-tags">
                  {result.skills_detail.missing_preferred.map((s) => (
                    <span key={s} className="badge badge-warning">{s}</span>
                  ))}
                </div>
              ) : (
                <p style={{ color: 'var(--text-secondary)', fontSize: 'var(--font-size-xs)' }}>✓ Matches all preferred qualifications.</p>
              )}
            </div>

            {/* Additional Skills */}
            <div className="dossier-card" style={{ borderTop: '3px solid var(--color-info)' }}>
              <h4 style={{ color: '#60a5fa', fontSize: 'var(--font-size-sm)' }}>
                + Additional Candidate Skills ({result.skills_detail.extra_skills.length})
              </h4>
              {result.skills_detail.extra_skills.length > 0 ? (
                <div className="skills-tags">
                  {result.skills_detail.extra_skills.map((s) => (
                    <span key={s} className="badge badge-info">{s}</span>
                  ))}
                </div>
              ) : (
                <p style={{ color: 'var(--text-tertiary)', fontSize: 'var(--font-size-xs)' }}>No additional skills listed.</p>
              )}
            </div>

          </div>
        </div>
      )}

      {/* TAB CONTENT: CANDIDATE PROFILE */}
      {activeTab === 'profile' && (
        <div className="dossier-grid animate-in">
          {/* Left: Bio, Experience, Projects */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)' }}>
            
            {/* Parsed Work History */}
            <div className="dossier-card">
              <h3 className="dossier-section-title"><span>💼</span> Parsed Work History</h3>
              {result.resume_data.experience && result.resume_data.experience.length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                  {result.resume_data.experience.map((exp, idx) => (
                    <div key={idx} style={{ borderBottom: idx < result.resume_data.experience.length - 1 ? '1px solid rgba(255,255,255,0.04)' : 'none', paddingBottom: 16 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8, marginBottom: 4 }}>
                        <strong style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-primary)' }}>
                          {exp.title || 'Role Position'}
                        </strong>
                        <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--accent-primary-light)', fontWeight: 600 }}>
                          {exp.duration || 'Date Range'}
                        </span>
                      </div>
                      <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)', marginBottom: 8, fontWeight: 500 }}>
                        🏢 {exp.company || 'Employer'}
                      </div>
                      {exp.description && (
                        <p style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                          {exp.description}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-tertiary)', textAlign: 'center', padding: '16px 0' }}>
                  No professional work experience parsed on the resume.
                </p>
              )}
            </div>

            {/* Personal & Academic Projects */}
            {result.resume_data.projects && result.resume_data.projects.length > 0 && (
              <div className="dossier-card">
                <h3 className="dossier-section-title"><span>🚀</span> Parsed Project Portfolio</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                  {result.resume_data.projects.map((proj, idx) => (
                    <div key={idx} style={{ background: 'rgba(255, 255, 255, 0.01)', padding: 12, borderRadius: 'var(--radius-sm)', border: '1px solid rgba(255, 255, 255, 0.03)' }}>
                      <div style={{ fontWeight: 600, color: 'var(--accent-primary-light)', fontSize: 'var(--font-size-sm)', marginBottom: 6 }}>
                        {proj.title}
                      </div>
                      {proj.description && (
                        <p style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)', marginBottom: 8, lineHeight: 1.4 }}>
                          {proj.description}
                        </p>
                      )}
                      {proj.technologies && proj.technologies.length > 0 && (
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                          {proj.technologies.map((tech) => (
                            <span key={tech} className="badge badge-neutral" style={{ fontSize: '9px', padding: '1px 6px' }}>
                              {tech}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

          </div>

          {/* Right: Contact info, education, Certifications, Strengths */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)' }}>
            
            {/* Bio Info */}
            <div className="dossier-card">
              <h3 className="dossier-section-title"><span>👤</span> Contact Dossier</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10, fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)' }}>
                <div>
                  <strong style={{ display: 'block', fontSize: 'var(--font-size-xs)', color: 'var(--text-tertiary)', textTransform: 'uppercase' }}>Candidate Name</strong>
                  <span style={{ color: 'var(--text-primary)' }}>{result.resume_data.name || 'Not extracted'}</span>
                </div>
                <div>
                  <strong style={{ display: 'block', fontSize: 'var(--font-size-xs)', color: 'var(--text-tertiary)', textTransform: 'uppercase' }}>Email Address</strong>
                  <span>📧 {result.resume_data.email || 'Not extracted'}</span>
                </div>
                <div>
                  <strong style={{ display: 'block', fontSize: 'var(--font-size-xs)', color: 'var(--text-tertiary)', textTransform: 'uppercase' }}>Phone Number</strong>
                  <span>📞 {result.resume_data.phone || 'Not extracted'}</span>
                </div>
                {result.resume_data.linkedin && (
                  <div>
                    <strong style={{ display: 'block', fontSize: 'var(--font-size-xs)', color: 'var(--text-tertiary)', textTransform: 'uppercase' }}>LinkedIn Profile</strong>
                    <a href={`https://${result.resume_data.linkedin}`} target="_blank" rel="noopener noreferrer">
                      🔗 {result.resume_data.linkedin}
                    </a>
                  </div>
                )}
                <div>
                  <strong style={{ display: 'block', fontSize: 'var(--font-size-xs)', color: 'var(--text-tertiary)', textTransform: 'uppercase' }}>Total Experience</strong>
                  <span style={{ color: 'var(--text-primary)' }}>💼 {result.resume_data.total_experience_years} years</span>
                </div>
              </div>
            </div>

            {/* Education History */}
            <div className="dossier-card">
              <h3 className="dossier-section-title"><span>🎓</span> Education Credentials</h3>
              {result.resume_data.education && result.resume_data.education.length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                  {result.resume_data.education.map((edu, i) => (
                    <div key={i} style={{ borderBottom: i < result.resume_data.education.length - 1 ? '1px solid rgba(255,255,255,0.04)' : 'none', paddingBottom: 8 }}>
                      <strong style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-primary)', display: 'block' }}>
                        {edu.degree}
                      </strong>
                      <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)', display: 'block', marginTop: 2 }}>
                        🏫 {edu.institution || 'School/College'} 
                        {edu.year && <> • Class of {edu.year}</>}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-tertiary)', textAlign: 'center' }}>No education data parsed.</p>
              )}
            </div>

            {/* Certifications */}
            {result.resume_data.certifications && result.resume_data.certifications.length > 0 && (
              <div className="dossier-card">
                <h3 className="dossier-section-title"><span>📜</span> Certifications</h3>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {result.resume_data.certifications.map((cert) => (
                    <span key={cert} className="badge badge-info" style={{ fontSize: '11px', padding: '4px 10px' }}>
                      {cert}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Major Strengths */}
            {result.strengths && result.strengths.length > 0 && (
              <div className="dossier-card">
                <h3 className="dossier-section-title"><span>💪</span> Candidate Strengths</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {result.strengths.map((str, idx) => (
                    <div key={idx} style={{ display: 'flex', gap: 8, alignItems: 'flex-start', fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)' }}>
                      <span style={{ color: 'var(--color-success)', fontWeight: 'bold' }}>✓</span>
                      <span>{str}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

          </div>
        </div>
      )}


      {/* Action Buttons */}
      <div
        style={{
          display: 'flex',
          gap: 12,
          justifyContent: 'center',
          marginTop: 'var(--space-xl)',
          paddingBottom: 'var(--space-2xl)',
        }}
      >
        <button
          className="btn btn-primary btn-lg"
          onClick={() => navigate('analyze')}
        >
          🔄 Screen Another Resume
        </button>
        <button
          className="btn btn-secondary btn-lg"
          onClick={() => window.print()}
        >
          🖨️ Print Screening Dossier
        </button>
      </div>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────

export default function App() {
  const { page, resultId, navigate } = useRouter();

  return (
    <>
      <Header page={page} navigate={navigate} />
      {page === 'home' && <HomePage navigate={navigate} />}
      {page === 'analyze' && <AnalyzePage navigate={navigate} />}
      {page === 'results' && resultId && (
        <ResultsPage resultId={resultId} navigate={navigate} />
      )}
    </>
  );
}
