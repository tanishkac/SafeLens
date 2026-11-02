import { ClusteringConfig, ClusteredHarmfulContent } from "@/utils/clustering";

export interface PolicyClauseCitation {
    clause_id: string;
    clause_name?: string;
    reason?: string;
}

export interface SuggestedEdit {
    action: string;
    target?: string;
    instruction?: string;
}

export interface HarmfulContent {
    startTime: number;
    endTime: number;
    confidence: number;
    type: string;
    description?: string;
    evidence?: string;
    source?: "vision" | "audio" | "ocr" | "multimodal";
    detectionMethod?: string;
    categories?: string[];
    factorWeights?: {
        visual?: number;
        audio?: number;
        text?: number;
    };
    citedPolicyClauses?: PolicyClauseCitation[];
    isEdsaExempt?: boolean;
    edsaRationale?: string;
    suggestedEdits?: SuggestedEdit[];
    rawData?: any; // Store original backend event data
}

export interface TranscriptionWord {
    word: string;
    start: number;
}

export interface TranscriptionLine {
    start: number;
    end: number;
    words: TranscriptionWord[];
    text: string;
}

export interface AnalysisData {
    summary: string;
    transcription: string;
    harmfulContent: HarmfulContent[]; // Keep original for backward compatibility
    clusteredHarmfulContent?: ClusteredHarmfulContent[]; // New clustered data
    analysisModel?: string;
    videoDuration?: number; // seconds, from backend metadata if available
    clusteringConfig?: ClusteringConfig;
    transcriptionWords?: TranscriptionWord[]; // Word-level timestamps for synced transcriptions
    transcriptionLines?: TranscriptionLine[]; // Optional cached line segmentation
}

export interface HarmfulContentMarker {
    time: number; // Maps to navigationTimestamp for VideoPlayer compatibility
    endTime: number;
    confidence: number;
    eventCount?: number;
    categories?: string[];
}

// Re-export clustering types for convenience
export type { ClusteringConfig, ClusteredHarmfulContent } from "@/utils/clustering";
