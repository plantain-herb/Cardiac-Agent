<div align="center">

# 🫀 BAAI Cardiac-Agent

**An Intelligent Cardiac MRI Analysis System Driven by a Multimodal Agent**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

The system orchestrates a fine-tuned LLaVA-based agent with multiple expert models to provide automated **sequence identification**, **cardiac structure segmentation**, **disease screening**, and **comprehensive report generation**.

</div>

---

## Demo

### Cardiac Structure Segmentation

Upload cardiac MRI and let the agent automatically identify sequences, select expert segmentation models, and return labeled results.

<video src="https://github.com/user-attachments/assets/53288ada-6634-4a6a-8393-3e9038de78ac" controls width="100%"></video>

### Disease Screening & Classification

The agent dispatches cardiac disease screening (CDS) and non-ischemic cardiomyopathy subtyping (NICMS) expert workers for automated diagnosis.

<video src="https://github.com/user-attachments/assets/2f60b850-68dc-41ae-a1f7-590c9968bc15" controls width="100%"></video>

### Report Generation

End-to-end pipeline: from raw DICOM/NIfTI uploads to a downloadable PDF report with cardiac metrics, classification results, and clinical evaluation.

<video src="https://github.com/user-attachments/assets/20d3b028-5c31-4e5e-abd7-107a8b925409" controls width="100%"></video>

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
| **Sequence Identification** | Automatically identifies CMR sequences (cine 2CH / 4CH / SA, LGE SA, Rest Myocardium Perfusion Imaging / Rest_MPI) via the Agent |
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
│                     FastAPI Backend (:8005)                         │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                    HeartMRIAgent                             │  │
│   │  • Sequence ID  • Smart Frame Extraction  • Modality Ordering│  │
│   │  • API Selection • Expert Dispatch • Result Summary          │  │
│   └─────────────────────┬────────────────────────────────────────┘  │
└─────────────────────────┼───────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────────┐
│                     Controller (:30000)                             │
│            Worker Registry · Heartbeat · Load Balancing             │
└──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬─────────────┘
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
├── data/                          # Agent test data
│   ├── api/                      # API selection test samples
│   └── findings/                 # Findings interpretation test samples
├── weights/
│   ├── agent/                    # HuggingFace layout
│   ├── cine_seg_first_2CH/       # Cine_2CH_seg1.pth
│   ├── cine_seg_second_2CH/      # Cine_2CH_seg2.pth
│   ├── cine_seg_first_4CH/       # Cine_4CH_seg1.pth
│   ├── cine_seg_second_4CH_L/    # Cine_4CH_seg2_L.pth
│   ├── cine_seg_second_4CH_R/    # Cine_4CH_seg2_R.pth
│   ├── cine_seg_first_SA/        # Cine_SAX_seg1.pth
│   ├── cine_seg_second_SA/       # Cine_SAX_seg2.pth
│   ├── lge_seg_first_SA/         # LGE_SAX_seg1.pth
│   ├── lge_seg_second_SA/        # LGE_SAX_seg2.pth
│   ├── diagnosis_first/          # CDS.pth
│   └── diagnosis_second/         # NICMS.pth
└── scripts/
    └── merge_lora_weights.py     # LoRA weight merging utility
```

## Model Weights

Place the agent under `weights/agent/` (HuggingFace layout). **Expert checkpoints** stay under the **traditional subdirectories** below; each folder holds the listed `.pth` filename (defaults in `app.config`: `expert_weight_path(subdir, filename)`).

| Subdirectory | Checkpoint file | Model | Task | Worker |
|---|---|---|---|---|
| `agent/` | *(HuggingFace files)* | LLaVA (LLaMA + Vision Encoder) | Multimodal Agent | `agent_worker.py` |
| `cine_seg_first_2CH/` | `Cine_2CH_seg1.pth` | Segmentation stage 1 | Cine 2-Chamber coarse | `cine_2ch_seg_worker.py` |
| `cine_seg_second_2CH/` | `Cine_2CH_seg2.pth` | Segmentation stage 2 | Cine 2-Chamber refined | `cine_2ch_seg_worker.py` |
| `cine_seg_first_4CH/` | `Cine_4CH_seg1.pth` | Segmentation stage 1 | Cine 4-Chamber coarse | `cine_4ch_seg_worker.py` |
| `cine_seg_second_4CH_L/` | `Cine_4CH_seg2_L.pth` | Segmentation stage 2 (left) | Cine 4-Chamber L refinement | `cine_4ch_seg_worker.py` |
| `cine_seg_second_4CH_R/` | `Cine_4CH_seg2_R.pth` | Segmentation stage 2 (right) | Cine 4-Chamber R refinement | `cine_4ch_seg_worker.py` |
| `cine_seg_first_SA/` | `Cine_SAX_seg1.pth` | Segmentation stage 1 | Cine short-axis coarse | `cine_sa_seg_worker.py` |
| `cine_seg_second_SA/` | `Cine_SAX_seg2.pth` | Segmentation stage 2 | Cine short-axis refined | `cine_sa_seg_worker.py` |
| `lge_seg_first_SA/` | `LGE_SAX_seg1.pth` | Segmentation stage 1 | LGE short-axis coarse | `lge_sa_seg_worker.py` |
| `lge_seg_second_SA/` | `LGE_SAX_seg2.pth` | Segmentation stage 2 | LGE short-axis refined | `lge_sa_seg_worker.py` |
| `diagnosis_first/` | `CDS.pth` | Classification | Cardiac Disease Screening | `cds_worker.py` |
| `diagnosis_second/` | `NICMS.pth` | Classification | NICMS subtyping | `nicms_worker.py` |

## Test Data

Download the evaluation set from Hugging Face: **[TaipingQu/CMRAgentEvalSet](https://huggingface.co/datasets/TaipingQu/CMRAgentEvalSet)**.
### Data Statistics
| View Folder | File Format | Number of Files | Total Z-axis Dimensions |
|-------------|-------------|-----------------|-------------------------|
| 2CH         | .nii.gz     | 203             | 16410                   |
| 4CH         | .nii.gz     | 205             | 16230                   |
| SA          | .nii.gz     | 206             | 58313                   |
| Rest_MPI    | .nii.gz     | 189             | 48391                   |
| LGE_SA      | .nii.gz     | 200             | 2131                    |
| Total       | .nii.gz     | 1003            | 141475                  |

After download, data is organized into two categories under the `data/` directory:

| Category | Directory | Description | Format |
|---|---|---|---|
| **2D API Selection** | `json/api_2D.json/` | Multi-turn conversation samples for 2D API selection: expert API choice, dispatch, and summarizing expert output back to the user | JSON |
| **3D API Selection** | `json/merged_vqa_cardiac_agent_API_3D.json/` | Multi-turn conversation samples for 3D API selection: expert API choice, dispatch, and summarizing expert output back to the user | JSON |
| **2D Findings Interpretation** | `json/findings_2D.json/` | Single-turn samples for direct image interpretation and clinical-style findings (e.g., valve status, chamber morphology) without calling any expert API | JSON |
| **3D Findings Interpretation** | `json/vqa_cardiac_agent_210_test_4CH_Cine_findings_3D.json/` | Single-turn samples for direct image interpretation and clinical-style findings focused on cardiac valve evaluation without calling any expert API | JSON |
| **3D Findings Interpretation** | `json/vqa_cardiac_agent_210_test_RestMPI_3D.json/` | Single-turn samples for direct image interpretation and clinical-style findings focused on myocardial perfusion assessment without calling any expert API | JSON |
| **3D Findings Interpretation** | `json/vqa_cardiac_agent_210_test_SAX_Cine_findings_3D.json/` | Single-turn samples for direct image interpretation and clinical-style findings focused on cardiac functional evaluation without calling any expert API | JSON |
| **3D Findings Interpretation** | `json/vqa_cardiac_agent_210_test_findings_3D.json/` | Single-turn samples for direct image interpretation and comprehensive overall CMR radiological assessment without calling any expert API | JSON |

Each JSON sample contains an `id`, a list of `image` paths, and multi-turn `conversations`. Each GPT turn includes structured `thoughts`, `actions` (API calls, if any), and a natural-language `value`.

<details>
<summary><b>API Selection example</b> — agent selects an expert API and processes its output</summary>

```json
{
  "id": "0001183_cine_4ch_slice_8",
  "image": [
    "image/cine/4ch/0001183_4ch_image/slice_8/slice_8_0.png",
    "image/cine/4ch/0001183_4ch_image/slice_8/slice_8_1.png",
    "image/cine/4ch/0001183_4ch_image/slice_8/slice_8_2.png"
  ],
  "conversations": [
    {
      "from": "human",
      "value": "Can you segment this image?\n<image>"
    },
    {
      "from": "gpt",
      "thoughts": "For this scan, a segmentation tool can help isolate the cardiac structures of interest.",
      "actions": [
        { "API_name": "4CH Cine Segmentation", "API_params": {} }
      ],
      "value": "Certainly! I have used the 4CH Cine Segmentation to delineate all four chambers in this four-chamber cine image for chamber size and morphology comparison."
    },
    {
      "from": "human",
      "value": "4CH Cine Segmentation output: {'message': 'The 4CH Cine Segmentation has processed the image.'}\n\nAnswer my first question: What technique can enhance the visualization of cardiac regions in this medical scan?"
    },
    {
      "from": "gpt",
      "thoughts": "Segmentation is done on this scan; I will provide the answer based on the output.",
      "actions": [],
      "value": "The result below provides a clear depiction, aiding in your medical evaluation."
    }
  ]
}
```

```json
{
    "id": "0000940_cc_cine_2ch_slice_10_cine_4ch_slice_2_cine_sa_slice_23",
    "image": [
      "image/nii/4ch/0000940_4ch_image.nii.gz",
      "image/nii/sa/0000940_sa_image.nii.gz",
      "image/nii/2ch/0000940_2ch_image.nii.gz"
    ],
    "conversations": [
      {
        "from": "human",
        "value": "Based on these cardiac MRI images, can you determine if this is normal heart function, ischemic cardiomyopathy, or non-ischemic cardiomyopathy?\n<image><image><image>"
      },
      {
        "from": "gpt",
        "thoughts": "Determining the type of cardiomyopathy requires analyzing the specific characteristics present in the image.",
        "actions": [
          {
            "API_name": "Cardiac Disease Screening",
            "API_params": {}
          }
        ],
        "value": "I will leverage the Cardiac Disease Screening tool to classify this as normal, ischemic, or non-ischemic cardiomyopathy."
      },
      {
        "from": "human",
        "value": "Cardiac Disease Screening output: Ischemic Cardiomyopathy\n\nAnswer my first question: Based on these cardiac MRI images, can you determine if this is normal heart function, ischemic cardiomyopathy, or non-ischemic cardiomyopathy?"
      },
      {
        "from": "gpt",
        "thoughts": "The Cardiac Disease Screening results enable me to provide a definitive answer about the cardiomyopathy type.",
        "actions": [],
        "value": "According to the Cardiac Disease Screening analysis results, the diagnosis is Ischemic Cardiomyopathy."
      }
    ]
  }
```

</details>

<details>
<summary><b>Findings Interpretation example</b> — agent directly reads the image and reports clinical findings</summary>

```json
{
  "id": "0001267_cine_4ch_slice_17",
  "image": [
    "image/cine/4ch/0001267_4ch_image/slice_17/slice_17_0.png",
    "image/cine/4ch/0001267_4ch_image/slice_17/slice_17_1.png",
    "image/cine/4ch/0001267_4ch_image/slice_17/slice_17_2.png"
  ],
  "conversations": [
    {
      "from": "human",
      "value": "Please evaluate the cardiac valve status in this image.\n<image>"
    },
    {
      "from": "gpt",
      "thoughts": "I can analyze the valve morphology and functional status directly from this image.",
      "actions": [],
      "value": "Mitral Valve: Normal; Tricuspid Valve: Normal"
    }
  ]
}
```
```json
{
  "id": "0001474-cine_2ch_slice_24-cine_4ch_slice_8-cine_sa_slice_21",
  "image": [
      "image/nii/4ch/0001474_4ch_image.nii.gz",
      "image/nii/sa/0001474_sa_image.nii.gz"
  ],
  "conversations": [
      {
          "from": "human",
          "value": "Can you identify the imaging findings from these images?\n<image><image>"
      },
      {
          "from": "gpt",
          "thoughts": "I can analyze medical images and provide insightful, clinically relevant responses.",
          "actions": [],
          "value": "Detailed analysis reveals thinning of the anterior and lateral walls of the left ventricle from the base to the apex, with overall weakened left ventricular wall motion and reduced systolic function. During systole,bundle-shaped hypointense blood flow extends toward the atrial side in the regions of the mitral and tricuspid valves; during diastole,bundle-shaped hypointense blood flow extends toward the ventricular side in the outflow tract area. A fluid signal intensity shadow is present within the pericardial cavity, suggesting possible pericardial effusion. Additionally, an arc-shaped fluid signal intensity shadow is observed in the left pleural cavity, indicating the presence of pleural effusion."
      }
  ]
}
```
</details>

### Agent Findings Categories

The agent's findings interpretation covers the following diagnostic items, each reported in `Key: Value` format:

| Group | Finding | Possible Values |
|---|---|---|
| **Valve Assessment** | Mitral Valve | Normal, Abnormal |
| | Tricuspid Valve | Normal, Abnormal |
| **Left Ventricle** | Left Ventricle | Normal, Enlarged, Dilated |
| | Left Ventricular Wall | Normal, Thickened, Thinned |
| | Left Ventricular Wall Motion | Normal, Reduced |
| | Left Ventricular Systolic Function | Normal, Decreased |
| | Left Ventricular Diastolic Function | Normal, Decreased |
| **Right Ventricle** | Right Ventricle | Normal, Enlarged, Dilated |
| | Right Ventricular Wall | Normal, Thickened, Thinned |
| | Right Ventricular Wall Motion | Normal, Reduced |
| | Right Ventricular Systolic Function | Normal, Decreased |
| | Right Ventricular Diastolic Function | Normal, Decreased |
| **Pericardium** | Pericardial | No Effusion, Effusion |
| **Perfusion** | Perfusion | Normal, Abnormal |
| **CINE Findings** | A brief description of the imaging findings |  |
| **LGE Findings** | A brief description of the imaging findings |  |
| **Rest_MPI Findings** | A brief description of the imaging findings |  |

## Tech Stack

- **Backend**: Python, FastAPI, Uvicorn
- **Deep Learning**: PyTorch, Hugging Face Transformers
- **Agent Model**: LLaVA (LLaMA + Vision Encoder), fine-tuned for cardiac MRI
- **Medical Imaging**: SimpleITK, NumPy
- **RAG / LLM**: OpenAI-compatible API (ChatCAD)
- **Frontend**: Vanilla HTML / CSS / JavaScript
- **Environment**: Conda (`cardiac_agent` for agent, `cardiac_models` for expert models)

## Prerequisites

- Python 3.8+
- CUDA-capable GPU(s) -- the agent and segmentation/classification workers require GPU
- Conda (for environment management)
- Model weights placed in `weights/`

## Quick Start

### 1. Weights & Data Download

| Resource | Hugging Face | Local path |
|---|---|---|
| **Model weights** | [TaipingQu/BAAI-Cardiac-Agent](https://huggingface.co/TaipingQu/BAAI-Cardiac-Agent/tree/main) (model) | `weights/` — `agent/` plus expert subdirs + `.pth` names in [Model Weights](#model-weights) |
| **Test data** | [TaipingQu/CMRAgentEvalSet](https://huggingface.co/datasets/TaipingQu/CMRAgentEvalSet) (dataset) | The supplementary dataset provides a comprehensive evaluation of CMR imaging findings, along with the corresponding API calls. |
| **CMR-MULTI** | [TaipingQu/CMR-MULTI](https://huggingface.co/datasets/TaipingQu/CMR-MULTI) (dataset) | CMR multi-sequence segmentation dataset |

### 2. Environment Setup

```bash
# Create and activate the agent environment
conda create -n cardiac_agent python=3.10
conda activate cardiac_agent
pip install -r requirements.txt

# Create and activate the expert models environment
conda create -n cardiac_models python=3.10
conda activate cardiac_models
pip install -r requirements_models.txt
```

### 3. Configuration

Copy `.env.example` to `.env` and set the required variables:

```bash
cp app/.env.example app/.env
```

Key environment variables:
- `API_KEY` -- API key for the LLM service (used by MIR / RAG)
- `API_BASE_URL` -- Base URL for the LLM API
- `MODEL` -- LLM model name (default: `deepseek-chat`)

### 4. Start All Services

```bash
# Start everything (controller → agent → workers → demo)
./app/start.sh full

# Or start components individually:
./app/start.sh controller    # Controller only
./app/start.sh agent         # LLaVA Agent only
./app/start.sh workers       # All expert workers
./app/start.sh demo          # Backend + Frontend
```

### 5. Access the Web Demo

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
| Controller | 30000 | cardiac_agent | - |
| LLaVA Agent | 40000 | cardiac_agent | Yes |
| Cine 2CH Segmentation | 21010 | cardiac_models | Yes |
| Cine 4CH Segmentation | 21011 | cardiac_models | Yes |
| Cine SA Segmentation | 21012 | cardiac_models | Yes |
| LGE SA Segmentation | 21013 | cardiac_models | Yes |
| Cardiac Disease Screening | 21020 | cardiac_models | Yes |
| NICMS Subtyping | 21021 | cardiac_models | Yes |
| Medical Report Generation | 21030 | cardiac_models | - |
| Cardiac Metrics | 21031 | cardiac_models | - |
| Medical Info Retrieval | 21040 | cardiac_agent | - |
| Sequence Analysis | 21050 | cardiac_agent | - |
| Demo Backend (API) | 8005 | cardiac_agent | - |
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
| `/api/download/{session_id}/{type}/{filename}` | GET  Download results (nifti / segmentation / reports) |
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
| Rest Myocardium Perfusion Imaging (Rest_MPI) | `tp` | Sequence identification only |

## Acknowledgements

The architecture of Cardiac-Agent is heavily inspired by **[MMedAgent](https://github.com/Wangyixinxin/MMedAgent)** (Li et al., EMNLP 2024). We adopt their Agent-Expert design pattern and extend it to the cardiac MRI domain with specialized expert workers for CMR sequence identification, multi-view segmentation, disease screening, and report generation.

We also gratefully acknowledge the following projects that Cardiac-Agent builds upon:

- [LLaVA](https://github.com/haotian-liu/LLaVA) / [LLaVA-Med](https://github.com/microsoft/LLaVA-Med) -- Multimodal large language model backbone
- [ChatCAD](https://github.com/zhaozh10/ChatCAD) -- RAG-based medical information retrieval

If you use our work, please also consider citing the following:

```bibtex
@misc{qu2026baaicardiacagentintelligent,
      title={BAAI Cardiac Agent: An intelligent multimodal agent for automated reasoning and diagnosis of cardiovascular diseases from cardiac magnetic resonance imaging}, 
      author={Taiping Qu and Hongkai Zhang and Lantian Zhang and Can Zhao and Nan Zhang and Hui Wang and Zhen Zhou and Mingye Zou and Kairui Bo and Pengfei Zhao and Xingxing Jin and Zixian Su and Kun Jiang and Huan Liu and Yu Du and Maozhou Wang and Ruifang Yan and Zhongyuan Wang and Tiejun Huang and Lei Xu and Henggui Zhang},
      year={2026},
      eprint={2604.04078},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2604.04078}, 
}
```

## License

Copyright 2026 Taiping Qu, Beijing Academy of Artificial Intelligence (BAAI)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
