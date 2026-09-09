import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

def analyze_drone_frame(image_bytes: bytes, user_prompt: str) -> str:
    """Analyzes a raw aerial frame using a Vision-Language Model."""
    encoded_img = base64.b64encode(image_bytes).decode('utf-8')
    vlm = ChatOpenAI(model="gpt-4o", max_tokens=350)
    
    system_prompt = (
        "You are an AI perimeter security engine analyzing drone imagery. "
        "Inspect the frame for structural damage, unauthorized vehicles, thermal anomalies, or perimeter breaches. "
        "Summarize observed objects, list detected risks, and assign a threat score from 0 (Safe) to 100 (Critical)."
    )
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": f"{system_prompt}\n\nTask: {user_prompt}"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_img}"}
            }
        ]
    )
    
    response = vlm.invoke([message])
    return response.content
