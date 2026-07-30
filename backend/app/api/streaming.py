"""
backend/app/api/streaming.py
----------------------------------------------------
GENKIT AI v5.0 Server-Sent Events (SSE) Token Stream Generator
Asynchronous token generator yielding standard text/event-stream data lines.
"""

import json
import asyncio
from typing import AsyncGenerator, Generator
from starlette.responses import StreamingResponse


class SSEEventStream:
    """Enterprise Server-Sent Events Stream Formatter."""

    @staticmethod
    async def format_sse_generator(
        token_generator: Generator[str, None, None],
        session_id: str,
        intent: str = "General",
        confidence: float = 1.0,
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronously yields formatted SSE stream events.
        """
        # Yield metadata start event
        meta_payload = {
            "event": "start",
            "session_id": session_id,
            "intent": intent,
            "confidence": confidence,
        }
        yield f"data: {json.dumps(meta_payload)}\n\n"
        await asyncio.sleep(0.01)

        # Yield stream token chunks
        for token in token_generator:
            token_payload = {"event": "token", "chunk": token}
            yield f"data: {json.dumps(token_payload)}\n\n"
            await asyncio.sleep(0.005)

        # Yield completion event
        end_payload = {"event": "end", "status": "completed"}
        yield f"data: {json.dumps(end_payload)}\n\n"


def create_sse_response(
    token_generator: Generator[str, None, None],
    session_id: str,
    intent: str = "General",
    confidence: float = 1.0,
) -> StreamingResponse:
    """Wraps SSE generator into Starlette StreamingResponse with correct event-stream headers."""
    async_gen = SSEEventStream.format_sse_generator(
        token_generator, session_id=session_id, intent=intent, confidence=confidence
    )
    return StreamingResponse(
        async_gen,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
