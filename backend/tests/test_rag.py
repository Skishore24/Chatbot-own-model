"""
backend/tests/test_rag.py
----------------------------------------------------
Unit tests for Hybrid RAG Pipeline, BM25, TF-IDF, and Grounding.
"""

import unittest
from app.rag.chunker import DocumentChunk
from app.rag.pipeline import HybridRAGPipeline
from app.rag.grounding import DOMAIN_REFUSAL_MESSAGE


class TestHybridRAG(unittest.TestCase):

    def setUp(self):
        self.sample_chunks = [
            DocumentChunk(
                id="services_web",
                source="services",
                category="Services",
                title="Web Development",
                text="Genkit builds high-performance responsive web applications using React, Next.js, FastAPI, and Node.js.",
                keywords=["web", "website", "react", "nextjs", "development"],
            ),
            DocumentChunk(
                id="services_mobile",
                source="services",
                category="Services",
                title="Mobile App Development",
                text="We engineer cross-platform mobile apps for iOS and Android using Flutter.",
                keywords=["mobile", "flutter", "ios", "android", "apps"],
            ),
            DocumentChunk(
                id="company_info",
                source="company",
                category="Company",
                title="Genkit Company Overview",
                text="Genkit is an enterprise software company founded by Kishore and Surya.",
                keywords=["company", "founders", "kishore", "surya", "about"],
            ),
        ]
        self.pipeline = HybridRAGPipeline(chunks=self.sample_chunks)

    def test_in_domain_retrieval(self):
        """Test in-domain query returns relevant document chunk."""
        chunks, confidence, is_grounded = self.pipeline.retrieve("What web technologies do you use?")
        self.assertTrue(is_grounded)
        self.assertGreater(confidence, 0.2)
        self.assertTrue(len(chunks) > 0)
        self.assertEqual(chunks[0].title, "Web Development")

    def test_out_of_domain_refusal(self):
        """Test out-of-domain query is correctly flagged and refused."""
        chunks, confidence, is_grounded = self.pipeline.retrieve("What is the capital of France?")
        self.assertFalse(is_grounded)
        self.assertEqual(self.pipeline.get_refusal_answer(), DOMAIN_REFUSAL_MESSAGE)

    def test_prompt_construction(self):
        """Verify prompt builder structure."""
        chunks, _, _ = self.pipeline.retrieve("Tell me about the founders")
        prompt = self.pipeline.build_prompt("Tell me about the founders", chunks)
        self.assertIn("SYSTEM:", prompt)
        self.assertIn("CONTEXT:", prompt)
        self.assertIn("QUESTION:", prompt)
        self.assertIn("ANSWER:", prompt)


if __name__ == "__main__":
    unittest.main()
