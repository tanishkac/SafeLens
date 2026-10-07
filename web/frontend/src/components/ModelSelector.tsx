"use client";

import { useState } from "react";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";

interface ModelSelectorProps {
    value?: string;
    onModelChange?: (model: string) => void;
}

const models = [
    { value: "google/gemini-flash-lite-latest", label: "Gemini Flash Lite (Google AI Studio)" },
    { value: "google/gemini-3.1-flash-lite", label: "Gemini 3.1 Flash Lite" },
    { value: "google/gemini-3-flash-preview", label: "Gemini 3 Flash (Preview)" },
    { value: "google/gemini-3.8-flash", label: "Gemini 3.8 Flash" },
    { value: "openrouter/free", label: "OpenRouter Free (Auto Vision/LLM)" },
    { value: "nvidia/nemotron-3.5-content-safety:free", label: "Nemotron Content Safety (Free)" },
    { value: "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", label: "Nemotron Omni Reasoning (Free)" },
    { value: "openai/gpt-4o-mini", label: "GPT-4o Mini" },
    { value: "anthropic/claude-3.5-haiku", label: "Claude 3.5 Haiku" },
    { value: "meta-llama/llama-3.1-8b-instruct", label: "Llama 3.1 8B" },
    { value: "qwen/qwen-2.5-7b-instruct", label: "Qwen 2.5 7B Instruct" },
    { value: "SafeLens/llama-3-8b", label: "Llama 3 8B (Local FT)" },
];

export default function ModelSelector({ value, onModelChange }: ModelSelectorProps) {
    const handleModelChange = (model: string) => {
        onModelChange?.(model);
    };

    return (
        <div className="flex items-center gap-3">
            <Label
                htmlFor="model-selector"
                className="text-sm font-medium whitespace-nowrap"
            >
                Analysis Model:
            </Label>
            <Select
                value={value || "google/gemini-flash-lite-latest"}
                onValueChange={handleModelChange}
            >
                <SelectTrigger id="model-selector" className="w-[200px]">
                    <SelectValue placeholder="Select model" />
                </SelectTrigger>
                <SelectContent>
                    {models.map((model) => (
                        <SelectItem key={model.value} value={model.value}>
                            {model.label}
                        </SelectItem>
                    ))}
                </SelectContent>
            </Select>
        </div>
    );
}
