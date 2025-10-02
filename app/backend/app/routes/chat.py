from fastapi import APIRouter, HTTPException
from app.models import ChatRequest, ChatResponse
from app.llm_chain import get_llm_response

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint for chatting with the LLM.
    """
    try:
        response_text = await get_llm_response(request.message)
        return ChatResponse(reply=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
