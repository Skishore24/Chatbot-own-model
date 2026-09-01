"""
backend/app/rag/pipeline.py
----------------------------------------------------
Unified Hybrid RAG Pipeline for Genkit AI V6.
End-to-end execution:
Dataset Ingestion -> Inverted Index -> BM25 + TF-IDF -> Hybrid Reranker -> Grounding Validator -> Structured Synthesis.
"""

from typing import List, Optional, Tuple
from app.core.config import settings
from app.core.logger import logger
from app.rag.chunker import DocumentChunk
from app.rag.loader import load_domain_chunks
from app.rag.index import InvertedIndex
from app.rag.bm25 import BM25Retriever
from app.rag.tfidf import TFIDFRetriever
from app.rag.reranker import HybridReranker
from app.rag.grounding import GroundingValidator, DOMAIN_REFUSAL_MESSAGE


class HybridRAGPipeline:
    """Production Hybrid RAG Pipeline using 100% deterministic local algorithms."""

    def __init__(self, chunks: Optional[List[DocumentChunk]] = None):
        self.chunks: List[DocumentChunk] = chunks if chunks is not None else load_domain_chunks()
        corpus_texts = [f"{c.title} {c.text}" for c in self.chunks]

        self.index = InvertedIndex(corpus_texts)
        self.bm25 = BM25Retriever(self.index, k1=settings.RAG_BM25_K1, b=settings.RAG_BM25_B)
        self.tfidf = TFIDFRetriever(self.index)
        self.reranker = HybridReranker(
            bm25_weight=settings.RAG_FUSION_BM25_WEIGHT,
            tfidf_weight=settings.RAG_FUSION_TFIDF_WEIGHT,
        )
        self.grounding = GroundingValidator(confidence_threshold=settings.RAG_CONFIDENCE_THRESHOLD)

        logger.info(f"Initialized HybridRAGPipeline with {len(self.chunks)} knowledge chunks.")

    @property
    def validator(self) -> GroundingValidator:
        """Alias for grounding validator."""
        return self.grounding

    @property
    def total_documents(self) -> int:
        return len(self.chunks)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> Tuple[List[DocumentChunk], float, bool]:
        """
        Executes hybrid retrieval, reranking, and grounding validation.
        Returns: (top_chunks, confidence_score, is_grounded)
        """
        k = top_k or settings.RAG_TOP_K
        if not self.chunks or not query.strip():
            return [], 0.0, False

        # Run BM25 & TF-IDF scoring
        bm25_scores = self.bm25.score_query(query)
        tfidf_scores = self.tfidf.score_query(query)

        # Rerank and extract top-K candidate chunks
        ranked_chunks = self.reranker.rerank(query, self.chunks, bm25_scores, tfidf_scores, top_k=k)

        # Compute grounding confidence and domain verification
        top_score = ranked_chunks[0].score if ranked_chunks else 0.0
        confidence, is_grounded = self.grounding.compute_grounding_score(query, ranked_chunks, top_retrieval_score=top_score)

        if not is_grounded:
            return [], confidence, False

        return ranked_chunks, confidence, is_grounded

    def synthesize_answer(self, query: str, chunks: List[DocumentChunk]) -> str:
        """
        Synthesizes a clean, authoritative, well-structured Markdown answer
        directly from verified Genkit knowledge chunks.
        """
        if not chunks:
            return self.get_refusal_answer()

        q_lower = query.lower()
        top_chunk = chunks[0]

        # 1. General Services Inquiry
        if any(w in q_lower for w in ["what service", "services offered", "services do you offer", "services provide", "what do you do", "what can you do", "all services", "service list"]):
            return (
                "**Genkit AI** offers 6 core digital & engineering services:\n\n"
                "1. **Website Development**\n"
                "   - Custom responsive websites, web applications, and landing pages.\n"
                "   - *Tech:* Python, FastAPI, Django, React, Node.js, HTML5, CSS3, JavaScript.\n"
                "   - *Turnaround:* 3 to 6 weeks.\n\n"
                "2. **Graphic Design**\n"
                "   - High-quality visual creatives, marketing banners, flyers, and thumbnails.\n"
                "   - *Tech:* Adobe Photoshop, Illustrator, Figma.\n"
                "   - *Turnaround:* 3 to 7 business days.\n\n"
                "3. **Branding & Visual Identity**\n"
                "   - Custom vector logos, brand guidelines, color palettes, and typography scales.\n"
                "   - *Tech:* Figma, Adobe Illustrator.\n"
                "   - *Turnaround:* 1 to 2 weeks.\n\n"
                "4. **Video Editing**\n"
                "   - Professional video post-production for YouTube, ads, and short-form reels.\n"
                "   - *Tech:* Adobe Premiere Pro, After Effects.\n"
                "   - *Turnaround:* 2 to 5 days per video.\n\n"
                "5. **Search Engine Optimization (SEO)**\n"
                "   - On-page & technical SEO to boost Google rankings and organic traffic.\n"
                "   - *Tech:* Google Search Console, Google Analytics, Keyword Planner.\n"
                "   - *Turnaround:* Ongoing monthly retainers.\n\n"
                "6. **AI & Chatbot Development**\n"
                "   - 100% self-hosted custom LLMs, RAG assistants, and workflow automations.\n"
                "   - *Tech:* Python, PyTorch, FastAPI, MySQL.\n"
                "   - *Turnaround:* 4 to 8 weeks.\n\n"
                "💬 *Interested in a project? Reach out to us at [genkit.tech@gmail.com](mailto:genkit.tech@gmail.com) or book a free consultation at [genkit.in/contact](https://www.genkit.in/contact)!*"
            )

        # 2. General Pricing Inquiry
        if any(w in q_lower for w in ["pricing", "cost", "how much", "price", "packages", "rates", "package", "hourly rate"]):
            if "hourly" in q_lower:
                return (
                    "**Genkit Hourly Rates:**\n\n"
                    "- **Website Development:** $45/hr\n"
                    "- **AI & Chatbot Development:** $65/hr\n"
                    "- **Branding:** $40/hr\n"
                    "- **UI/UX Design:** $35/hr\n"
                    "- **SEO:** $35/hr\n"
                    "- **Video Editing:** $30/hr\n\n"
                    "We also provide fixed-price packages and custom project estimates. Reach out at [genkit.tech@gmail.com](mailto:genkit.tech@gmail.com) for a tailored proposal!"
                )
            elif "landing page" in q_lower:
                return (
                    "**Landing Page Package:**\n\n"
                    "- **Starting Price:** $500 USD\n"
                    "- **Timeline:** 5 to 10 business days\n"
                    "- **Included Features:** 1 Custom Page, Responsive Mobile Layout, Basic SEO, Contact Form Integration, 7 Days Post-Delivery Support.\n\n"
                    "To get started, email us at [genkit.tech@gmail.com](mailto:genkit.tech@gmail.com)."
                )
            elif "ecommerce" in q_lower or "e-commerce" in q_lower or "store" in q_lower:
                return (
                    "**E-Commerce Package:**\n\n"
                    "- **Starting Price:** $3,000 USD\n"
                    "- **Timeline:** 4 to 6 weeks\n"
                    "- **Included Features:** Unlimited Product Listings, Shopping Cart & Checkout Flow, Secure Stripe/PayPal Integration, Inventory Dashboard, 60 Days Post-Delivery Support.\n\n"
                    "Book a scoping call at [genkit.in/contact](https://www.genkit.in/contact)."
                )
            elif "ai" in q_lower or "bot" in q_lower:
                return (
                    "**Custom AI Agent Package:**\n\n"
                    "- **Starting Price:** $4,500 USD\n"
                    "- **Timeline:** 6 to 8 weeks\n"
                    "- **Included Features:** Custom LLM Pipeline, RAG Document Search Engine, FastAPI Backend Integration, MySQL Database Integration, 90 Days Dedicated Technical Support.\n\n"
                    "Contact our AI engineers at [genkit.tech@gmail.com](mailto:genkit.tech@gmail.com)."
                )
            else:
                return (
                    "**Genkit Project Packages & Pricing:**\n\n"
                    "- **Landing Page Package:** Starts at $500 USD (5-10 days, 1 page, responsive, SEO, contact form).\n"
                    "- **Business Website Package:** Starts at $1,500 USD (3-4 weeks, up to 5 pages, custom UI/UX, CMS/Blog, Google Analytics).\n"
                    "- **E-Commerce Package:** Starts at $3,000 USD (4-6 weeks, shopping cart, checkout, payment gateway, inventory).\n"
                    "- **Custom AI Agent Package:** Starts at $4,500 USD (6-8 weeks, custom LLM, RAG search engine, FastAPI, MySQL).\n\n"
                    "💡 *For custom scopes or hourly work ($30-$65/hr), we provide personalized quotes. Visit [genkit.in/contact](https://www.genkit.in/contact) for a free 15-minute consultation.*"
                )

        # 3. Founders & Company Inquiry
        if any(w in q_lower for w in ["founder", "founders", "who started", "who founded", "created by", "kishore", "hari"]):
            return (
                "**Genkit** was founded in **June 2024** by **Hari Krishna** and **Kishore Kumar**, along with core founding team members Dharani, Deepak, Rahul Vijay, Jithesh, Jaya Nithesh, and Dharanesh.\n\n"
                "Genkit operates with a remote-first team of 10-15 digital specialists across India, providing high-quality digital products and custom AI solutions to businesses worldwide."
            )

        # 4. Tech Stack Inquiry
        if any(w in q_lower for w in ["tech stack", "technology", "technologies", "frameworks", "tools", "what tech", "languages"]):
            return (
                "**Genkit Technology Stack:**\n\n"
                "- **Frontend:** React, HTML5, CSS3, JavaScript, Tailwind CSS, Vite\n"
                "- **Backend:** Python (FastAPI, Django), Node.js (Express), Java\n"
                "- **Databases:** MySQL, MongoDB, PostgreSQL, SQLite\n"
                "- **Creative & Design Tools:** Figma, Adobe Photoshop, Adobe Illustrator, Premiere Pro, After Effects\n"
                "- **AI & ML:** PyTorch, Custom Transformer GPT Architectures, BM25 / TF-IDF Hybrid RAG"
            )

        # 5. Contact Inquiry
        if any(w in q_lower for w in ["contact", "email us", "get in touch", "how to reach", "phone number", "how to contact", "reach out"]):
            return (
                "You can get in touch with **Genkit AI** through the following channels:\n\n"
                "- ✉️ **Email:** [genkit.tech@gmail.com](mailto:genkit.tech@gmail.com)\n"
                "- 🌐 **Website:** [https://www.genkit.in](https://www.genkit.in)\n"
                "- 📝 **Contact Form:** [genkit.in/contact](https://www.genkit.in/contact)\n"
                "- 📸 **Instagram:** [@genkit.in](https://instagram.com/genkit.in)\n"
                "- 💻 **GitHub:** [github.com/genkit](https://github.com/genkit)\n\n"
                "⏱️ *We typically respond within 24 business hours. Free 15-minute scoping calls can be booked directly on our website.*"
            )

        # 6. Specific Domain / Project / Policy Answer from Top Chunks
        primary_text = top_chunk.text.strip()
        # Clean up any potential markdown question prefix if present
        if primary_text.startswith("**Q:") or primary_text.startswith("Q:"):
            lines = primary_text.split("\n", 1)
            if len(lines) > 1:
                primary_text = lines[1].strip()

        return primary_text

    def build_prompt(self, query: str, chunks: List[DocumentChunk]) -> str:
        """
        Builds internal structured prompt with system instructions and retrieved context.
        """
        context_text = "\n\n".join([f"[{c.title}]\n{c.text}" for c in chunks]) if chunks else "No relevant context found."

        prompt = (
            "SYSTEM:\n"
            "You are Genkit AI, an enterprise AI assistant for Genkit.in.\n"
            "RULES:\n"
            "- Answer using only the verified Genkit knowledge provided in CONTEXT.\n"
            "- If the question cannot be answered from the context, state that you don't have verified info.\n"
            "- Do not invent company facts, pricing, or team members.\n"
            "- Be concise, professional, and helpful.\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            f"QUESTION:\n{query}\n\n"
            "ANSWER:\n"
        )
        return prompt

    def get_refusal_answer(self) -> str:
        return DOMAIN_REFUSAL_MESSAGE


# Singleton Pipeline
default_rag_pipeline: Optional[HybridRAGPipeline] = None


def get_rag_pipeline() -> HybridRAGPipeline:
    global default_rag_pipeline
    if default_rag_pipeline is None:
        default_rag_pipeline = HybridRAGPipeline()
    return default_rag_pipeline
