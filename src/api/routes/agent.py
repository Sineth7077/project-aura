"""Agent decision routes.

WebSocket endpoint for real-time gesture streaming:

- Clients send observation payloads (gesture events as JSON).
- The server processes them through a full agent pipeline (observation →
  context → decision → permission → execution → verification).
- Responses are structured AgentDecideResponse JSON.
- This is a WebSocket-based implementation, NOT production WebRTC.
- No physical or sensitive actions execute without explicit permission.
"""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json

from src.api.schemas.agent import AgentDecideRequest, AgentDecideResponse
from src.api.services.agent_service import AgentService

router = APIRouter(tags=["agent"])

agent_service: AgentService = AgentService()


@router.post("/agent/decide", response_model=AgentDecideResponse)
async def agent_decide(request: AgentDecideRequest) -> AgentDecideResponse:
    """Accept a validated observation and return a complete decision trace."""
    request_data = {
        "observation_source": request.observation_source,
        "observation_type": request.observation_type,
        "data": request.data,
        "confidence": request.confidence,
        "context": request.context,
    }
    result = agent_service.decide(request_data)
    return AgentDecideResponse(
        decision_id=result["decision_id"],
        selected_action=result["selected_action"],
        confidence=result["confidence"],
        explanation=result["explanation"],
        permission_status=result["permission_status"],
        permission_reason=result.get("permission_reason"),
        execution_status=result["execution_status"],
        verification_status=result["verification_status"],
        timestamp=result["timestamp"],
        execution_result=result.get("execution_result"),
        verification_record=result.get("verification_record"),
    )


@router.websocket("/ws/gesture")
async def gesture_stream(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time gesture event streaming.

    Protocol:
    - Client sends JSON: {"observation_type": "gesture", "data": {"gesture": "OPEN_PALM"}, "confidence": 0.9}
    - Server responds with structured decision JSON including complete pipeline status.
    - All actions go through permission checks and verification.
    """
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON payload",
                }))
                continue

            # Process through the complete agent pipeline
            result = agent_service.decide({
                "observation_source": "websocket",
                "observation_type": payload.get("observation_type", "gesture"),
                "data": payload.get("data", {}),
                "confidence": payload.get("confidence"),
            })
            response = {
                "type": "decision",
                "decision_id": result["decision_id"],
                "selected_action": result["selected_action"],
                "confidence": result["confidence"],
                "explanation": result["explanation"],
                "permission_status": result["permission_status"],
                "permission_reason": result.get("permission_reason"),
                "execution_status": result["execution_status"],
                "verification_status": result["verification_status"],
                "timestamp": result["timestamp"],
            }
            await websocket.send_text(json.dumps(response))
    except WebSocketDisconnect:
        pass
