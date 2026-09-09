from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from langchain_core.messages import HumanMessage
from app.graph import sentinel_agent

app = FastAPI(title="Autonomous Aerial Perimeter Sentinel")

@app.post("/v1/sentinel/process")
async def process_telemetry(
    prompt: str = Form(...),
    frame: UploadFile | None = File(None)
):
    image_bytes = None
    if frame:
        image_bytes = await frame.read()
        
    try:
        initial_state = {
            "messages": [HumanMessage(content=prompt)],
            "image_data": image_bytes,
            "route": "",
            "verified": False
        }
        
        result = sentinel_agent.invoke(initial_state)
        final_answer = result["messages"][-1].content
        
        return {
            "status": "completed",
            "execution_route": result.get("route"),
            "verified_safety": result.get("verified"),
            "response": final_answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
