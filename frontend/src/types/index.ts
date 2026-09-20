/* ── TypeScript Type Definitions ─────────────────────────────────────────── */

export interface GradeInfo {
  grade: string;
  label: string;
  color: string;
}

export interface CategoryScores {
  skills_match: number;
  experience_match: number;
  education_match: number;
  semantic_similarity: number;
}

export interface EducationEntry {
  degree: string;
  level?: string;
  institution?: string;
  year?: string;
}

export interface ExperienceEntry {
  title?: string;
  company?: string;
  duration?: string;
  description?: string;
}

export interface ProjectEntry {
  title: string;
  description?: string;
  technologies: string[];
}

export interface ResumeGap {
  category: string;
  missing_item: string;
  impact: 'high' | 'medium' | 'low';
  suggestion: string;
}

export interface ResumeData {
  name?: string;
  email?: string;
  phone?: string;
  linkedin?: string;
  skills: string[];
  skills_categorized: Record<string, string[]>;
  education: EducationEntry[];
  experience: ExperienceEntry[];
  projects: ProjectEntry[];
  certifications: string[];
  total_experience_years: number;
  is_student: boolean;
}

export interface JobDescriptionData {
  title?: string;
  all_skills: string[];
  required_skills: string[];
  preferred_skills: string[];
  min_experience_years?: number;
  education_requirements: string[];
}

export interface SkillsDetail {
  matched_skills: string[];
  missing_required: string[];
  missing_preferred: string[];
  extra_skills: string[];
}

export interface Recommendation {
  priority: 'high' | 'medium' | 'low';
  category: string;
  text: string;
}

export interface ATSSuggestion {
  category: string;
  text: string;
}

export interface ChecklistItem {
  criteria: string;
  status: string;
  details: string;
  is_met: boolean;
}

export interface HiringAssessment {
  verdict: string;
  verdict_color: string;
  leveling: string;
  summary: string;
  checklist: ChecklistItem[];
  screening_actions: string[];
}

export interface AnalysisResult {
  id: string;
  created_at: string;
  status: string;
  overall_score: number;
  grade: GradeInfo;
  category_scores: CategoryScores;
  skills_detail: SkillsDetail;
  strengths: string[];
  recommendations: Recommendation[];
  negatives: ResumeGap[];
  ats_suggestions: ATSSuggestion[];
  hiring_assessment?: HiringAssessment;
  resume_data: ResumeData;
  job_data: JobDescriptionData;
  resume_filename: string;
  resume_file_type: string;
}

export interface AnalysisListItem {
  id: string;
  created_at: string;
  status: string;
  overall_score: number;
  grade: GradeInfo;
  resume_filename: string;
  job_title?: string;
  matched_skills_count: number;
  missing_skills_count: number;
  category_scores: CategoryScores;
}

export interface AnalysisListResponse {
  analyses: AnalysisListItem[];
  total: number;
}
