import axios from 'axios';
import type { AnalysisResult, AnalysisListResponse } from '../types';

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 120000, // 2 minutes — NLP analysis can be slow on first run
});

/**
 * Upload a resume and job description for AI analysis.
 */
export async function analyzeResume(
  file: File,
  jobDescription: string,
): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append('resume', file);
  formData.append('job_description', jobDescription);

  const response = await api.post<AnalysisResult>('/analyze', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });

  return response.data;
}

/**
 * Fetch list of past analyses.
 */
export async function getAnalyses(
  skip = 0,
  limit = 20,
): Promise<AnalysisListResponse> {
  const response = await api.get<AnalysisListResponse>('/analyses', {
    params: { skip, limit },
  });
  return response.data;
}

/**
 * Fetch a single analysis by ID.
 */
export async function getAnalysis(id: string): Promise<AnalysisResult> {
  const response = await api.get<AnalysisResult>(`/analyses/${id}`);
  return response.data;
}

/**
 * Delete an analysis.
 */
export async function deleteAnalysis(id: string): Promise<void> {
  await api.delete(`/analyses/${id}`);
}
