"""
SafeLens Policy RAG (Retrieval-Augmented Generation) Engine.

Indexes official YouTube Community Guidelines and retrieves the most relevant
policy clauses for multimodal video segments based on audio transcripts,
OCR on-screen text, and visual frame descriptions.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
POLICIES_FILE = BASE_DIR / "data" / "policies" / "youtube_guidelines.json"
EMBEDDINGS_CACHE_FILE = BASE_DIR / "data" / "policies" / "policy_embeddings.npy"


class PolicyRAGEngine:
    """
    Retrieval-Augmented Generation engine for YouTube Community Guidelines.
    
    Uses all-MiniLM-L6-v2 (sentence-transformers) with cosine similarity ranking
    and cached precomputed policy embeddings for sub-millisecond retrieval.
    Includes robust TF-IDF fallback if transformer embeddings are unavailable.
    """
    _instance: Optional["PolicyRAGEngine"] = None

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        auto_index: bool = True
    ):
        self.model_name = model_name
        self.policies: List[Dict[str, Any]] = []
        self.policy_map: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.encoder = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self._edsa_overview = ""

        if auto_index:
            self.load_and_index()

    @classmethod
    def get_instance(cls) -> "PolicyRAGEngine":
        """Singleton accessor to avoid re-initializing encoder across requests."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_and_index(self) -> None:
        """Load policy JSON and prepare semantic vector index."""
        if not POLICIES_FILE.exists():
            logger.error(f"Policy file missing at {POLICIES_FILE}")
            return

        with open(POLICIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.policies = data.get("policies", [])
        self._edsa_overview = data.get("edsa_framework_overview", "")
        self.policy_map = {p["clause_id"]: p for p in self.policies}

        # Build comprehensive text representation for each policy
        corpus = [self._build_policy_document(p) for p in self.policies]

        # 1. Try local sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            try:
                # Prefer local cache to prevent external network calls
                self.encoder = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                self.encoder = SentenceTransformer(self.model_name)

            # Check if precomputed embeddings exist and match corpus size
            if EMBEDDINGS_CACHE_FILE.exists():
                try:
                    cached_emb = np.load(str(EMBEDDINGS_CACHE_FILE))
                    if cached_emb.shape[0] == len(self.policies):
                        self.embeddings = cached_emb
                        logger.info(f"Loaded {len(self.policies)} policy embeddings from cache: {EMBEDDINGS_CACHE_FILE.name}")
                        return
                except Exception as e:
                    logger.warning(f"Failed to load cached embeddings: {e}, recomputing...")

            # Compute embeddings
            logger.info(f"Encoding {len(self.policies)} policy clauses with {self.model_name}...")
            self.embeddings = self.encoder.encode(corpus, normalize_embeddings=True)
            try:
                np.save(str(EMBEDDINGS_CACHE_FILE), self.embeddings)
                logger.info(f"Saved policy embeddings cache to {EMBEDDINGS_CACHE_FILE.name}")
            except Exception as e:
                logger.warning(f"Could not persist embeddings cache: {e}")

        except Exception as e:
            logger.warning(f"sentence-transformers initialization failed ({e}). Falling back to TF-IDF.")
            self._init_tfidf_fallback(corpus)

    def _init_tfidf_fallback(self, corpus: List[str]) -> None:
        """Fallback to scikit-learn TF-IDF cosine ranking if neural encoder is unavailable."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            logger.info("Initialized TF-IDF policy retrieval fallback.")
        except Exception as e:
            logger.error(f"Failed to initialize TF-IDF fallback: {e}")

    @staticmethod
    def _build_policy_document(policy: Dict[str, Any]) -> str:
        """Constructs an information-dense text document for embedding."""
        triggers_str = ", ".join(policy.get("triggers", []))
        return (
            f"Category: {policy.get('category', '')}. "
            f"Subclause: {policy.get('subclause', '')}. "
            f"Official Title: {policy.get('official_title', '')}. "
            f"Summary: {policy.get('summary', '')} "
            f"Visual and Audio Triggers: {triggers_str}. "
            f"EDSA Exceptions: {policy.get('edsa_exceptions', '')}. "
            f"Remediation: {policy.get('suggested_action', '')}"
        )

    def retrieve(
        self,
        query_text: str,
        top_k: int = 3,
        category_filter: Optional[str] = None,
        min_similarity: float = 0.15
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant YouTube Community Guideline clauses.

        Args:
            query_text: Multimodal evidence text (Audio transcript, OCR, and Frame captions).
            top_k: Number of policy clauses to retrieve (default: 3).
            category_filter: Optional category name to restrict retrieval.
            min_similarity: Minimum similarity score threshold.

        Returns:
            List of policy clause dictionaries enriched with 'relevance_score'.
        """
        if not self.policies or not query_text or not query_text.strip():
            return []

        clean_query = query_text.strip()

        # Method 1: Semantic Embedding Retrieval (Primary)
        if self.encoder is not None and self.embeddings is not None:
            try:
                q_emb = self.encoder.encode([clean_query], normalize_embeddings=True)[0]
                # Cosine similarity is simply dot product because vectors are normalized
                scores = np.dot(self.embeddings, q_emb)

                # Filter and rank
                ranked_indices = np.argsort(scores)[::-1]
                results = []

                for idx in ranked_indices:
                    score = float(scores[idx])
                    if score < min_similarity and len(results) >= 1:
                        # Ensure we return at least 1 top result if requested, but stop if score is too low
                        break

                    candidate = self.policies[idx]
                    if category_filter and candidate.get("category", "").lower() != category_filter.lower():
                        continue

                    item = dict(candidate)
                    item["relevance_score"] = round(score, 4)
                    results.append(item)

                    if len(results) >= top_k:
                        break

                return results
            except Exception as e:
                logger.error(f"Semantic retrieval failed: {e}. Falling back to TF-IDF.")

        # Method 2: TF-IDF Cosine Retrieval (Fallback)
        if self.tfidf_vectorizer is not None and self.tfidf_matrix is not None:
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                q_vec = self.tfidf_vectorizer.transform([clean_query])
                sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]
                ranked_indices = np.argsort(sims)[::-1][:top_k]

                results = []
                for idx in ranked_indices:
                    score = float(sims[idx])
                    candidate = self.policies[idx]
                    if category_filter and candidate.get("category", "").lower() != category_filter.lower():
                        continue
                    item = dict(candidate)
                    item["relevance_score"] = round(score, 4)
                    results.append(item)
                return results
            except Exception as e:
                logger.error(f"TF-IDF retrieval error: {e}")

        # Method 3: Keyword heuristic fallback
        return self._keyword_search(clean_query, top_k)

    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Dead-simple fallback keyword search."""
        q_lower = query.lower()
        scored = []
        for p in self.policies:
            score = 0.0
            for trigger in p.get("triggers", []):
                if trigger.lower() in q_lower:
                    score += 0.35
            if p.get("category", "").lower() in q_lower:
                score += 0.20
            if p.get("subclause", "").lower() in q_lower:
                score += 0.20
            scored.append((score, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for s, p in scored[:top_k]:
            item = dict(p)
            item["relevance_score"] = round(s, 2)
            results.append(item)
        return results

    def get_policy(self, clause_id: str) -> Optional[Dict[str, Any]]:
        """Lookup policy details by clause_id (e.g. 'YT-HATE-02')."""
        return self.policy_map.get(clause_id)

    def format_for_prompt(self, retrieved_policies: List[Dict[str, Any]]) -> str:
        """
        Format retrieved policies into a concise, structured markdown prompt block
        for LLM safety reasoning.
        """
        if not retrieved_policies:
            return "No specific YouTube policy clauses matched. Use general safety guidelines."

        lines = [
            "APPLICABLE YOUTUBE COMMUNITY GUIDELINES (RETRIEVED VIA POLICY RAG):",
            "------------------------------------------------------------------"
        ]
        for idx, p in enumerate(retrieved_policies, 1):
            clause_id = p.get("clause_id", f"CLAUSE-{idx}")
            title = p.get("official_title", p.get("category", "Policy"))
            subclause = p.get("subclause", "")
            summary = p.get("summary", "")
            edsa = p.get("edsa_exceptions", "No standard exceptions.")
            action = p.get("suggested_action", "Review segment.")

            lines.append(
                f"[{idx}] CLAUSE {clause_id}: {title} ({subclause})\n"
                f"    Rule Summary: {summary}\n"
                f"    EDSA Exceptions: {edsa}\n"
                f"    Recommended Action: {action}"
            )

        return "\n".join(lines)


# Convenient module-level functions
_global_engine: Optional[PolicyRAGEngine] = None


def get_rag_engine() -> PolicyRAGEngine:
    """Retrieve global singleton engine instance."""
    global _global_engine
    if _global_engine is None:
        _global_engine = PolicyRAGEngine.get_instance()
    return _global_engine


def retrieve_relevant_policies(query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Helper function to retrieve policy clauses for a segment."""
    return get_rag_engine().retrieve(query_text, top_k=top_k)


def format_retrieved_policies(policies: List[Dict[str, Any]]) -> str:
    """Helper function to format policies for LLM prompt."""
    return get_rag_engine().format_for_prompt(policies)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n--- Testing SafeLens Policy RAG Engine ---")
    engine = PolicyRAGEngine()

    test_queries = [
        "Swastika banner hanging on the wall during political demonstration",
        "Street fight with a man getting kicked repeatedly in the head with visible blood",
        "Documentary discussing Adolf Hitler and Nazi war crimes in World War 2 museum context",
        "Pointing a modified handgun with a Glock full-auto switch directly at the camera",
    ]

    for q in test_queries:
        print(f"\nQUERY: \"{q}\"")
        matches = engine.retrieve(q, top_k=2)
        for m in matches:
            print(f"  -> [{m['clause_id']}] {m['category']} - {m['subclause']} (Score: {m['relevance_score']})")
            print(f"     Suggested Fix: {m['suggested_action']}")
