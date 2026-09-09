# Autonomous Aerial Perimeter Sentinel 

An end-to-end, stateful multi-agent system built for real-time drone telemetry analysis, automated visual perimeter sweeps, and operational SOP compliance checking. 

The platform uses **LangGraph** to coordinate routing, **Vision-Language Models (VLMs)** for aerial anomaly detection, and **Pinecone** for low-latency vector retrieval—all exposed via an asynchronous **FastAPI** service and instrumented with **LangSmith** for observability.

---

## Key Features

* **Stateful Multi-Agent Workflow:** Built with **LangGraph** to support dynamic query routing, perception tool execution, and output verification.
* **VLM Aerial Anomaly Detection:** Processes raw aerial drone frames using GPT-4 Vision to detect structural flaws, unauthorized perimeter breaches, or thermal hazards.
* **Low-Latency Vector Memory:** Integrates **Pinecone** vector database to ground operational responses in real-time safety and regulatory SOPs[cite: 3].
* **LangSmith Observability:** Traces execution graphs, token consumption, and confidence scores for every node step[cite: 3].
* **Production-Ready REST API:** Asynchronous **FastAPI** backend supporting multi-part payload processing (form parameters + binary images).
* **Cloud-Native Containerization:** Dockerized and pre-configured for automated deployment on **GCP Cloud Run** or **AWS Lambda/EC2**[cite: 3].

---

## System Architecture

```text
                                  ┌───────────────────────────┐
                                  │      Client / Telemetry    │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │      FastAPI Backend      │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │   LangGraph Router Node   │
                                  └───────┬───────────────┬───┘
                                          │               │
                         [Aerial Image]   │               │   [SOP / Policy Query]
                                          ▼               ▼
                        ┌──────────────────┐     ┌──────────────────┐
                        │ Vision Agent Node│     │ RAG Vector Node  │
                        │   (GPT-4 Vision) │     │ (Pinecone Index) │
                        └────────┬─────────┘     └────────┬─────────┘
                                 │                        │
                                 └───────────┬────────────┘
                                             │
                                             ▼
                                ┌───────────────────────────┐
                                │   Reflection / Verifier   │
                                └─────────────┬─────────────┘
                                              │
                                              ▼
                                ┌───────────────────────────┐
                                │     LangSmith Tracing     │
                                └───────────────────────────┘

``` 
## Tech StackOrchestration: 

- Python 3.11, LangGraph, LangChain  Vector 
- Storage: Pinecone (Serverless)
- Vision & LLM Models: OpenAI GPT-4o (VLM), text-embedding-3-small
- API Backend: FastAPI, Uvicorn, Pydantic
- Observability: LangSmith
- Deployment & Containerization: Docker, GCP Cloud Run 

## Repository Structure

aerial-perimeter-sentinel/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application routes
│   ├── graph.py         # LangGraph graph definition & routing logic
│   ├── vlm_engine.py    # VLM visual processing node
│   └── pinecone_rag.py  # Pinecone retrieval node
├── Dockerfile           # Production Container setup
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation


# Getting Started
## Prerequisites
- Python 3.11+
- Docker Desktop (optional, for containerized local execution)
- OpenAI API Key & Pinecone API Key

## 1. Environment Setup
Clone the repository and install the required dependencies:

```text
git clone [https://github.com/Neelamjaahnavi/autonomous-aerial-sentinel.git](https://github.com/Neelamjaahnavi/autonomous-aerial-sentinel.git)
cd autonomous-aerial-sentinel

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

```

## 2. Configure Environment Variables
Create a .env file in the project root:

```text
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=drone-sentinel-sops

# LangSmith Observability
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="[https://api.smith.langchain.com](https://api.smith.langchain.com)"
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=Aerial-Perimeter-Sentinel
```

## 3. Run Application Locally
Launch the FastAPI execution server:

```text
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Access the interactive API documentation at: http://localhost:8000/docs

# API Usage
```text POST /v1/sentinel/process```
Submits query text along with an optional aerial drone frame image for analysis.

Example Request (curl):
```text
curl -X 'POST' \
  'http://localhost:8000/v1/sentinel/process' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'prompt=Inspect the perimeter wall in this frame for unauthorized vehicles or damage.' \
  -F 'frame=@/path/to/aerial_frame.jpg;type=image/jpeg'
```
Example Response:

```text
{
  "status": "completed",
  "execution_route": "vision",
  "verified_safety": true,
  "response": "Perimeter Inspection Summary:\n1. Objects Detected: Utility truck parked along West Fence Segment B.\n2. Risk Assessment: Vehicle lacks authorization markers.\n3. Threat Level: Medium (Score: 45/100). Recommend dispatching ground security to verify vehicle credentials."
}
```

# Docker & Cloud Deployment

Local Docker Run
```text
docker build -t aerial-sentinel .
docker run -d -p 8080:8080 --env-file .env aerial-sentinel
```

Deploy to GCP Cloud Run
```text
gcloud run deploy aerial-sentinel \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=$OPENAI_API_KEY,PINECONE_API_KEY=$PINECONE_API_KEY
```
# License
Distributed under the MIT License. See LICENSE for details.
