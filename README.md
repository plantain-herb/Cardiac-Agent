<div align="center">

# 🫀 Cardiac-Agent

**An Intelligent Cardiac MRI Analysis System Driven by a Multimodal Agent**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-red.svg)]()

The system orchestrates a fine-tuned LLaVA-based agent with multiple expert models to provide automated **sequence identification**, **cardiac structure segmentation**, **disease screening**, and **comprehensive report generation**.

</div>

---

## Demo

### Cardiac Structure Segmentation

Upload cardiac MRI and let the agent automatically identify sequences, select expert segmentation models, and return labeled results.

<video src="[demo/seg.mp4](https://github.com/plantain-herb/Cardiac-Agent/blame/main/demo/cls.mp4)" controls width="100%"></video>

### Disease Screening & Classification

The agent dispatches cardiac disease screening (CDS) and non-ischemic cardiomyopathy subtyping (NICMS) expert workers for automated diagnosis.

<video src="demo/cls.mp4" controls width="100%"></video>

### Report Generation

End-to-end pipeline: from raw DICOM/NIfTI uploads to a downloadable PDF report with cardiac metrics, classification results, and clinical evaluation.

<video src="demo/mrg.mp4" controls width="100%"></video>

---

## Overview

Cardiac-Agent adopts an **Agent-Expert** architecture: a central multimodal agent (based on LLaVA) interprets user queries and cardiac MRI images, selects the appropriate expert API, dispatches tasks to specialized deep-learning workers, and summarizes the results in natural language.

**Core pipeline:**

```
Upload (DICOM / NIfTI) → Agent Sequence Identification → Smart Frame Extraction
→ Modality Ordering → Agent API Selection → Expert Worker Execution → Agent Summary
```

## Key Features

| Feature | Description |
|---|---|
| **Sequence Identification** | Automatically identifies CMR sequences (cine 2CH / 4CH / SA, LGE SA, T1/T2 mapping) via the Agent |
| **Cardiac Segmentation** | Multi-view segmentation: cine 2-chamber, 4-chamber, short-axis, and LGE short-axis |
| **Disease Screening (CDS)** | Three-class classification: Normal / Ischemic Cardiomyopathy / Non-ischemic Cardiomyopathy |
| **Cardiomyopathy Subtyping (NICMS)** | Five-class subtyping: HCM / DCM / Inflammatory / Restrictive / Arrhythmogenic |
| **Cardiac Metrics** | Quantitative analysis: LV/RV ejection fraction, volumes, stroke volume, cardiac output, myocardial mass, 17-segment wall thickness |
| **Report Generation (MRG)** | Automated comprehensive cardiac evaluation report (PDF) integrating metrics, CDS, and NICMS results |
| **Medical Info Retrieval (MIR)** | RAG-based medical knowledge retrieval for clinical questions |
| **Agent VQA** | Direct visual question answering on cardiac MRI without calling expert workers |
| **Web Demo** | Interactive web interface for file upload, chat-based analysis, and report download |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Web Frontend (:8080)                        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│                     FastAPI Backend (:8005)                          │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                    HeartMRIAgent                              │  │
│   │  • Sequence ID  • Smart Frame Extraction  • Modality Ordering│  │
│   │  • API Selection • Expert Dispatch • Result Summary          │  │
│   └─────────────────────┬────────────────────────────────────────┘  │
└─────────────────────────┼──────────────────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────────────────┐
│                    Controller (:30000)                               │
│            Worker Registry · Heartbeat · Load Balancing              │
└──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬────────────┘
       │      │      │      │      │      │      │      │
   ┌───▼──┐┌──▼──┐┌──▼──┐┌──▼──┐┌──▼──┐┌──▼──┐┌──▼──┐┌──▼──┐
   │Agent ││Seg  ││Seg  ││Seg  ││Seg  ││ CDS ││NICMS││ MRG │ ...
   │LLaVA ││2CH  ││4CH  ││ SA  ││ LGE ││     ││     ││     │
   │:40000││:2101││:2101││:2101││:2101││:2102││:2102││:2103│
   │(GPU) ││0    ││1    ││2    ││3    ││0    ││1    ││0    │
   └──────┘└─────┘└─────┘└─────┘└─────┘└─────┘└─────┘└─────┘
```

## Project Structure

```
Cardiac-Agent/
├── app/                          # Application layer
│   ├── config.py                 # Central configuration (URLs, mappings, paths)
│   ├── server.py                 # FastAPI entry point (uvicorn)
│   ├── start.sh                  # Service orchestration script
│   ├── routes/
│   │   └── api.py                # REST API routes (session, upload, chat, download)
│   ├── services/
│   │   ├── heart_agent.py        # Core HeartMRIAgent orchestration logic
│   │   ├── agent_client.py       # LLaVA Agent HTTP client
│   │   ├── expert_client.py      # Expert worker HTTP client
│   │   ├── seq_identifier.py     # Parallel sequence identification
│   │   └── session_manager.py    # Session & cache management
│   ├── utils/
│   │   ├── dicom.py              # DICOM/NIfTI I/O, frame extraction
│   │   ├── conversation.py       # Conversation persistence (JSON)
│   │   └── report.py             # PDF report generation
│   └── frontend/                 # Web UI (HTML/CSS/JS)
│       ├── index.html
│       ├── app.js
│       └── style.css
├── demo/                         # Demo videos
│   ├── demo_segmentation.mp4     # Segmentation demo
│   ├── demo_disease_screening.mp4# Disease screening demo
│   └── demo_report.mp4           # Report generation demo
├── serve/                        # Worker services
│   ├── controller.py             # Worker registry & dispatch controller
│   ├── agent_worker.py           # LLaVA multimodal agent worker (GPU)
│   ├── cine_2ch_seg_worker.py    # Cine 2-chamber segmentation (GPU)
│   ├── cine_4ch_seg_worker.py    # Cine 4-chamber segmentation (GPU)
│   ├── cine_sa_seg_worker.py     # Cine short-axis segmentation (GPU)
│   ├── lge_sa_seg_worker.py      # LGE short-axis segmentation (GPU)
│   ├── cds_worker.py             # Cardiac Disease Screening (GPU)
│   ├── nicms_worker.py           # Non-ischemic cardiomyopathy subtyping (GPU)
│   ├── metrics_worker.py         # Cardiac metrics calculation (CPU)
│   ├── mrg_worker.py             # Medical Report Generation orchestrator (CPU)
│   ├── mir_worker.py             # Medical Info Retrieval / RAG (CPU)
│   └── seq_worker.py             # Sequence analysis via Agent (CPU)
├── llava/                        # LLaVA multimodal model
│   ├── model/                    # Model architecture (LLaMA backbone + vision)
│   ├── train/                    # Multi-image training scripts
│   └── eval/                     # VQA evaluation
├── src/                          # Expert model source code
│   ├── CINE_2CH_SEG/             # 2-chamber segmentation model
│   ├── CINE_4CH_SEG/             # 4-chamber segmentation model
│   ├── CINE_SA_SEG/              # Short-axis segmentation model
│   ├── LGE_SA_SEG/               # LGE short-axis segmentation model
│   ├── CDS/                      # Cardiac Disease Screening model
│   ├── NICMS/                    # Non-ischemic cardiomyopathy model
│   ├── CMR/                      # Cardiac metrics calculation
│   └── RAG/ChatCAD/              # RAG for medical info retrieval
├── weights/                      # Model weights directory
│   └── agent/                    # Fine-tuned LLaVA agent weights
└── scripts/
    └── merge_lora_weights.py     # LoRA weight merging utility
```

## Tech Stack

- **Backend**: Python, FastAPI, Uvicorn
- **Deep Learning**: PyTorch, Hugging Face Transformers
- **Agent Model**: LLaVA (LLaMA + Vision Encoder), fine-tuned for cardiac MRI
- **Medical Imaging**: SimpleITK, NumPy
- **RAG / LLM**: OpenAI-compatible API (ChatCAD)
- **Frontend**: Vanilla HTML / CSS / JavaScript
- **Environment**: Conda (`mmedagent` for agent, `cardiac_models` for expert models)

## Prerequisites

- Python 3.8+
- CUDA-capable GPU(s) -- the agent and segmentation/classification workers require GPU
- Conda (for environment management)
- Model weights placed in `weights/`

## Quick Start

### 1. Environment Setup

```bash
# Create and activate the agent environment
conda create -n mmedagent python=3.10
conda activate mmedagent
pip install -r requirements.txt

# Create and activate the expert models environment
conda create -n cardiac_models python=3.10
conda activate cardiac_models
pip install -r requirements_models.txt
```

### 2. Configuration

Copy `.env.example` to `.env` and set the required variables:

```bash
cp app/.env.example app/.env
```

Key environment variables:
- `API_KEY` -- API key for the LLM service (used by MIR / RAG)
- `API_BASE_URL` -- Base URL for the LLM API
- `MODEL` -- LLM model name (default: `deepseek-chat`)

### 3. Start All Services

```bash
# Start everything (controller → agent → workers → demo)
./app/start.sh full

# Or start components individually:
./app/start.sh controller    # Controller only
./app/start.sh agent         # LLaVA Agent only
./app/start.sh workers       # All expert workers
./app/start.sh demo          # Backend + Frontend
```

### 4. Access the Web Demo

Open your browser and navigate to `http://localhost:8080`.

### Service Management

```bash
./app/start.sh status        # Check service status
./app/start.sh health        # HTTP health checks
./app/start.sh stop          # Stop all services
./app/start.sh restart       # Restart all services
```

## Service Ports

| Service | Port | Environment | GPU |
|---|---|---|---|
| Controller | 30000 | mmedagent | - |
| LLaVA Agent | 40000 | mmedagent | Yes |
| Cine 2CH Segmentation | 21010 | cardiac_models | Yes |
| Cine 4CH Segmentation | 21011 | cardiac_models | Yes |
| Cine SA Segmentation | 21012 | cardiac_models | Yes |
| LGE SA Segmentation | 21013 | cardiac_models | Yes |
| Cardiac Disease Screening | 21020 | cardiac_models | Yes |
| NICMS Subtyping | 21021 | cardiac_models | Yes |
| Medical Report Generation | 21030 | cardiac_models | - |
| Cardiac Metrics | 21031 | cardiac_models | - |
| Medical Info Retrieval | 21040 | mmedagent | - |
| Sequence Analysis | 21050 | mmedagent | - |
| Demo Backend (API) | 8005 | mmedagent | - |
| Demo Frontend | 8080 | - | - |

## API Reference

### Unified Chat Endpoint

```
POST /api/chat
```

The primary interface. Accepts multimodal inputs (DICOM ZIP, NIfTI, PNG images) and natural language questions. The agent automatically determines the appropriate analysis pipeline.

**Form parameters:**
- `message` -- User question (string)
- `files` -- Uploaded files (multipart, supports `.zip`, `.nii.gz`, `.nii`, `.png`, `.jpg`)
- `session_id` -- Session ID (optional, auto-created if empty)
- `task_type` -- Imaging modality: `mr`, `ct`, `us`, `ecg` (default: `mr`)

### Specialized Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/segment` | POST | Direct segmentation (single file) |
| `/api/classify` | POST | Direct classification (multiple files) |
| `/api/session/create` | POST | Create a new session |
| `/api/session/{id}/files` | GET | List uploaded files |
| `/api/session/{id}/frames` | GET | Get extracted frames |
| `/api/download/{session_id}/{type}/{filename}` | GET | Download results (nifti / segmentation / reports) |
| `/api/conversation/{session_id}` | GET | List conversation records |
| `/health` | GET | Health check |

## Supported Input Formats

- **DICOM** -- ZIP archive containing a DICOM series
- **NIfTI** -- `.nii` or `.nii.gz` volume files
- **Images** -- `.png`, `.jpg`, `.jpeg` for direct Agent VQA

## Supported CMR Sequences

| Sequence | Modality Key | Supported Tasks |
|---|---|---|
| Cine 2-Chamber | `cine_2ch` | Segmentation, CDS (optional) |
| Cine 4-Chamber | `cine_4ch` | Segmentation, CDS, NICMS (optional), MRG |
| Cine Short-Axis | `cine_sa` | Segmentation, CDS, NICMS, MRG, Metrics |
| LGE Short-Axis | `lge_sa` | Segmentation, NICMS, MRG (optional) |
| T1 / T2 Mapping | `tp` | Sequence identification only |

## License

All rights reserved.
