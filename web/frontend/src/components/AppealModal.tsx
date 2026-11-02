"use client";

import React, { useState, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Copy, Check, Download, X, Scale, ShieldAlert, Sparkles, RefreshCw } from "lucide-react";

interface PolicyClause {
    clause_id: string;
    clause_name?: string;
    reason?: string;
}

interface Remediation {
    action: string;
    target?: string;
    instruction?: string;
}

interface AppealData {
    video_id: string;
    title: string;
    appeal_letter_markdown: string;
    flagged_clauses: PolicyClause[];
    edsa_claim_applicable: boolean;
    edsa_summary: string;
    proposed_remediations: Remediation[];
    appeal_strength: string;
}

interface AppealModalProps {
    videoId: string;
    isOpen: boolean;
    onClose: () => void;
}

export default function AppealModal({ videoId, isOpen, onClose }: AppealModalProps) {
    const [data, setData] = useState<AppealData | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [copied, setCopied] = useState(false);

    const fetchAppealDraft = async () => {
        setLoading(true);
        setError(null);
        try {
            const res = await fetch(`/api/analyze/${videoId}/appeal-draft`);
            if (!res.ok) {
                // Try fallback URL
                const fallbackRes = await fetch(`/api/videos/${videoId}/appeal-draft`);
                if (!fallbackRes.ok) {
                    throw new Error(`Failed to generate appeal draft: ${res.statusText}`);
                }
                const result = await fallbackRes.json();
                setData(result);
                return;
            }
            const result = await res.json();
            setData(result);
        } catch (err: any) {
            setError(err.message || "Failed to load appeal draft");
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (isOpen && videoId) {
            fetchAppealDraft();
        }
    }, [isOpen, videoId]);

    const handleCopy = () => {
        if (!data?.appeal_letter_markdown) return;
        navigator.clipboard.writeText(data.appeal_letter_markdown);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    const handleDownload = () => {
        if (!data?.appeal_letter_markdown) return;
        const blob = new Blob([data.appeal_letter_markdown], { type: "text/markdown;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `YouTube_Appeal_${videoId}.md`;
        link.click();
        URL.revokeObjectURL(url);
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-xs p-4 animate-in fade-in duration-200">
            <Card className="w-full max-w-4xl max-h-[90vh] flex flex-col shadow-2xl border-primary/20 bg-background overflow-hidden">
                <CardHeader className="flex flex-row items-center justify-between pb-3 border-b">
                    <div className="flex items-center gap-2">
                        <Scale className="h-6 w-6 text-primary" />
                        <div>
                            <CardTitle className="text-xl flex items-center gap-2">
                                Official YouTube Appeal Letter Generator
                                <Badge variant="outline" className="text-xs bg-primary/10 text-primary border-primary/30">
                                    <Sparkles className="h-3 w-3 mr-1 inline" /> Policy RAG
                                </Badge>
                            </CardTitle>
                            <p className="text-xs text-muted-foreground mt-0.5">
                                Automated legal defense citing YouTube Community Guidelines & EDSA Exceptions
                            </p>
                        </div>
                    </div>
                    <Button variant="ghost" size="icon" onClick={onClose} className="rounded-full">
                        <X className="h-5 w-5" />
                    </Button>
                </CardHeader>

                <CardContent className="flex-1 min-h-0 flex flex-col p-6 space-y-4">
                    {loading && (
                        <div className="flex-1 flex flex-col items-center justify-center py-12 space-y-3">
                            <RefreshCw className="h-8 w-8 animate-spin text-primary" />
                            <p className="text-sm font-medium">Retrieving policy clauses & drafting defensible appeal...</p>
                            <p className="text-xs text-muted-foreground">Evaluating Educational, Documentary, Scientific, or Artistic (EDSA) exemptions</p>
                        </div>
                    )}

                    {error && (
                        <div className="p-4 rounded-lg bg-destructive/10 text-destructive text-sm flex items-center gap-2">
                            <ShieldAlert className="h-5 w-5 shrink-0" />
                            <span>{error}</span>
                            <Button variant="outline" size="sm" onClick={fetchAppealDraft} className="ml-auto">
                                Retry
                            </Button>
                        </div>
                    )}

                    {!loading && data && (
                        <>
                            {/* Metadata & Strength Bar */}
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 p-3 bg-muted/40 rounded-lg text-xs">
                                <div>
                                    <span className="font-semibold text-muted-foreground">Video Title:</span>
                                    <p className="font-medium text-foreground truncate">{data.title || videoId}</p>
                                </div>
                                <div>
                                    <span className="font-semibold text-muted-foreground">Appeal Defense Basis:</span>
                                    <p className="font-medium text-foreground">{data.edsa_summary}</p>
                                </div>
                                <div>
                                    <span className="font-semibold text-muted-foreground">Estimated Viability:</span>
                                    <p className="font-bold text-emerald-600 dark:text-emerald-400">{data.appeal_strength}</p>
                                </div>
                            </div>

                            {/* Cited Clauses Tags */}
                            {data.flagged_clauses && data.flagged_clauses.length > 0 && (
                                <div className="space-y-1">
                                    <span className="text-xs font-semibold text-muted-foreground">Cited YouTube Clauses in Defense:</span>
                                    <div className="flex flex-wrap gap-1.5">
                                        {data.flagged_clauses.map((c, i) => (
                                            <Badge key={i} variant="secondary" className="text-xs font-mono">
                                                [{c.clause_id}] {c.clause_name || "Guideline"}
                                            </Badge>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Letter Scroll Area */}
                            <div className="flex-1 min-h-0 border rounded-lg overflow-hidden bg-card">
                                <ScrollArea className="h-[48vh] w-full p-4 font-sans text-sm leading-relaxed whitespace-pre-wrap select-text">
                                    {data.appeal_letter_markdown}
                                </ScrollArea>
                            </div>

                            {/* Actions Footer */}
                            <div className="flex items-center justify-between pt-2 border-t">
                                <div className="text-xs text-muted-foreground flex items-center gap-1.5">
                                    <ShieldAlert className="h-3.5 w-3.5 text-amber-500" />
                                    <span>Copy directly into <strong>YouTube Studio &gt; Content &gt; Appeal Decision</strong></span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <Button variant="outline" size="sm" onClick={handleDownload} className="text-xs">
                                        <Download className="h-3.5 w-3.5 mr-1.5" />
                                        Download .md
                                    </Button>
                                    <Button size="sm" onClick={handleCopy} className="text-xs">
                                        {copied ? (
                                            <>
                                                <Check className="h-3.5 w-3.5 mr-1.5 text-emerald-400" />
                                                Copied to Clipboard!
                                            </>
                                        ) : (
                                            <>
                                                <Copy className="h-3.5 w-3.5 mr-1.5" />
                                                Copy Appeal Letter
                                            </>
                                        )}
                                    </Button>
                                </div>
                            </div>
                        </>
                    )}
                </CardContent>
            </Card>
        </div>
    );
}
