#!/bin/bash
#
# 心脏医学影像Agent系统 - 服务启动脚本
#
# 使用方法:
#   ./app/start.sh all          # 启动所有服务
#   ./app/start.sh controller   # 只启动Controller
#   ./app/start.sh workers      # 只启动所有Workers
#   ./app/start.sh demo         # 启动Demo (Backend API + Frontend)
#   ./app/start.sh frontend     # 只启动前端 (Frontend only)
#   ./app/start.sh full         # 启动完整系统 (所有服务 + Demo)
#   ./app/start.sh stop         # 停止所有服务
#   ./app/start.sh status       # 查看服务状态
#

set -e

# 配置 — PROJECT_DIR 始终指向 MMedAgent 根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 路径统一来自 app/config.py（单一事实源）
# 如果 Python 可用则动态读取，否则使用与 config.py 一致的默认值
_py_cfg() { python3 -c "from app.config import $1; print($1)" 2>/dev/null; }
LOG_DIR="$(_py_cfg LOG_DIR || echo "${PROJECT_DIR}/logs")"
PID_DIR="$(_py_cfg PID_DIR || echo "${PROJECT_DIR}/pids")"
FRONTEND_PID_DIR="$(_py_cfg FRONTEND_PID_DIR || echo "${PROJECT_DIR}/app/frontend/.pids")"

# Demo配置 — 前端移至 app/frontend, 后端用 python -m app.server
DEMO_DIR="${PROJECT_DIR}/app/frontend"
DEMO_BACKEND_PORT=8005
DEMO_FRONTEND_PORT=8080

# ============ Conda 环境配置 ============
CONDA_ENV_AGENT="mmedagent"           # LLaVA Agent模型环境
CONDA_ENV_EXPERT="cardiac_models"     # Expert模型环境 (分割/分类)
CONDA_ENV_DEMO="mmedagent"           # Demo前后端环境

# Conda初始化路径 (根据你的系统修改)
CONDA_PATH="${HOME}/anaconda3"

# ============ GPU 配置 ============
GPU_AGENT=0

GPU_SEG_2CH=0
GPU_SEG_4CH=0
GPU_SEG_SA=0
GPU_SEG_LGE=0

GPU_CDS=0
GPU_NICMS=0

# ============ 端口配置 ============
PORT_CONTROLLER=30000
PORT_AGENT=40000

PORT_SEG_2CH=21010
PORT_SEG_4CH=21011
PORT_SEG_SA=21012
PORT_SEG_LGE=21013

PORT_CDS=21020
PORT_NICMS=21021

PORT_MRG=21030
PORT_METRICS=21031
PORT_MIR=21040
PORT_SEQ=21050

# Agent模型路径 — 从 app/config.py 读取，保持单一事实源
AGENT_MODEL_PATH="$(_py_cfg AGENT_MODEL_PATH || echo "${PROJECT_DIR}/weights/agent")"

# 创建目录（与 config.py 中 os.makedirs 保持一致）
mkdir -p "${LOG_DIR}/workers"
mkdir -p "${PID_DIR}"
mkdir -p "${FRONTEND_PID_DIR}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_success() { echo -e "${BLUE}[SUCCESS]${NC} $1"; }

print_banner() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}          ${GREEN}🫀 Cardiac Agent System${NC}                              ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC}  Unified Agent-Driven Cardiac Imaging Analysis               ${BLUE}║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

activate_conda() {
    local env_name=$1
    source "${CONDA_PATH}/etc/profile.d/conda.sh"
    conda activate "${env_name}"
}

# ============ 启动各项服务 ============

start_controller() {
    log_info "启动 Controller (port ${PORT_CONTROLLER}, env: ${CONDA_ENV_AGENT})..."
    cd "${PROJECT_DIR}"

    nohup bash -c "
        source ${CONDA_PATH}/etc/profile.d/conda.sh
        conda activate ${CONDA_ENV_AGENT}
        python -m serve.controller \
            --host 0.0.0.0 \
            --port ${PORT_CONTROLLER} \
            --dispatch-method shortest_queue
    " > "${LOG_DIR}/controller.log" 2>&1 &

    echo $! > "${PID_DIR}/controller.pid"
    log_info "Controller 已启动 (PID: $!)"
    sleep 2
}

start_agent() {
    log_info "启动 LLaVA Agent模型 (port ${PORT_AGENT}, GPU ${GPU_AGENT}, env: ${CONDA_ENV_AGENT})..."
    cd "${PROJECT_DIR}"
    export CUDA_LAUNCH_BLOCKING=1

    CUDA_VISIBLE_DEVICES=${GPU_AGENT} nohup bash -c "
        source ${CONDA_PATH}/etc/profile.d/conda.sh
        conda activate ${CONDA_ENV_AGENT}
        cd ${PROJECT_DIR}
        python -m serve.agent_worker \
            --host 0.0.0.0 \
            --controller-address http://localhost:${PORT_CONTROLLER} \
            --port ${PORT_AGENT} \
            --worker-address http://localhost:${PORT_AGENT} \
            --model-path ${AGENT_MODEL_PATH} \
            --device cuda
    " > "${LOG_DIR}/agent_model.log" 2>&1 &

    echo $! > "${PID_DIR}/agent_model.pid"
    log_info "LLaVA Agent模型 已启动 (PID: $!, GPU: ${GPU_AGENT})"
    sleep 30
}

start_worker() {
    local name=$1
    local port=$2
    local module=$3
    local gpu=${4:-0}
    local cuda_device=${5:-$gpu}

    log_info "启动 ${name} Worker (port ${port}, CUDA ${cuda_device}, env: ${CONDA_ENV_EXPERT})..."
    cd "${PROJECT_DIR}"

    CUDA_VISIBLE_DEVICES=${cuda_device} nohup bash -c "
        source ${CONDA_PATH}/etc/profile.d/conda.sh
        conda activate ${CONDA_ENV_EXPERT}
        cd ${PROJECT_DIR}
        python -m serve.${module} \
            --host 0.0.0.0 \
            --port ${port} \
            --worker-address http://localhost:${port} \
            --controller-address http://localhost:${PORT_CONTROLLER} \
            --gpu ${gpu}
    " > "${LOG_DIR}/workers/${name}.log" 2>&1 &

    echo $! > "${PID_DIR}/${name}.pid"
    log_info "${name} 已启动 (PID: $!, CUDA: ${cuda_device})"
    sleep 30
}

start_metrics_worker() {
    local name="metrics"
    local port=${PORT_METRICS}

    log_info "启动 ${name} Worker (port ${port}, CPU only, env: ${CONDA_ENV_EXPERT})..."
    cd "${PROJECT_DIR}"

    nohup bash -c "
        source ${CONDA_PATH}/etc/profile.d/conda.sh
        conda activate ${CONDA_ENV_EXPERT}
        cd ${PROJECT_DIR}
        python -m serve.metrics_worker \
            --host 0.0.0.0 \
            --port ${port} \
            --worker-address http://localhost:${port} \
            --controller-address http://localhost:${PORT_CONTROLLER} \
            --model-names CardiacMetricsCalculation \
            --worker-id metrics-worker
    " > "${LOG_DIR}/workers/${name}.log" 2>&1 &

    echo $! > "${PID_DIR}/${name}.pid"
    log_info "${name} 已启动 (PID: $!, CPU only)"
    sleep 5
}

start_mrg_worker() {
    local name="mrg"
    local port=${PORT_MRG}

    log_info "启动 ${name} Worker (port ${port}, CPU only, env: ${CONDA_ENV_EXPERT})..."
    cd "${PROJECT_DIR}"

    nohup bash -c "
        source ${CONDA_PATH}/etc/profile.d/conda.sh
        conda activate ${CONDA_ENV_EXPERT}
        cd ${PROJECT_DIR}
        python -m serve.mrg_worker \
            --host 0.0.0.0 \
            --port ${port} \
            --worker-address http://localhost:${port} \
            --controller-address http://localhost:${PORT_CONTROLLER} \
            --model-names MedicalReportGeneration \
            --worker-id mrg-worker \
            --metrics-worker-url http://localhost:${PORT_METRICS} \
            --cds-worker-url http://localhost:${PORT_CDS} \
            --nicms-worker-url http://localhost:${PORT_NICMS}
    " > "${LOG_DIR}/workers/${name}.log" 2>&1 &

    echo $! > "${PID_DIR}/${name}.pid"
    log_info "${name} 已启动 (PID: $!, CPU only)"
    sleep 5
}

start_mir_worker() {
    local name="mir"
    local port=${PORT_MIR}

    log_info "启动 ${name} Worker (port ${port}, CPU only, env: ${CONDA_ENV_EXPERT})..."
    cd "${PROJECT_DIR}"

    nohup bash -c "
        source ${CONDA_PATH}/etc/profile.d/conda.sh
        conda activate ${CONDA_ENV_EXPERT}
        cd ${PROJECT_DIR}
        python -m serve.mir_worker \
            --host 0.0.0.0 \
            --port ${port} \
            --worker-address http://localhost:${port} \
            --controller-address http://localhost:${PORT_CONTROLLER} \
            --model-names MedicalInformationRetrieval \
            --limit-model-concurrency 5
    " > "${LOG_DIR}/workers/${name}.log" 2>&1 &

    echo $! > "${PID_DIR}/${name}.pid"
    log_info "${name} 已启动 (PID: $!, CPU only)"
    sleep 5
}

start_seq_worker() {
    local name="seq"
    local port=${PORT_SEQ}

    log_info "启动 ${name} Worker (port ${port}, CPU only, env: ${CONDA_ENV_EXPERT})..."
    cd "${PROJECT_DIR}"

    nohup bash -c "
        source ${CONDA_PATH}/etc/profile.d/conda.sh
        conda activate ${CONDA_ENV_EXPERT}
        cd ${PROJECT_DIR}
        python -m serve.seq_worker \
            --host 0.0.0.0 \
            --port ${port} \
            --worker-address http://localhost:${port} \
            --controller-address http://localhost:${PORT_CONTROLLER} \
            --agent-url http://localhost:${PORT_AGENT} \
            --model-names SequenceAnalysis \
            --limit-model-concurrency 5
    " > "${LOG_DIR}/workers/${name}.log" 2>&1 &

    echo $! > "${PID_DIR}/${name}.pid"
    log_info "${name} 已启动 (PID: $!, CPU only, depends on Agent)"
    sleep 5
}

start_workers() {
    log_info "启动所有 Expert Workers..."
    log_info "GPU分配: Cine2CHSeg=${GPU_SEG_2CH}, Cine4CHSeg=${GPU_SEG_4CH}, CineSASeg=${GPU_SEG_SA}, LgeSASeg=${GPU_SEG_LGE}, CDS=${GPU_CDS}, NICMS=${GPU_NICMS}"

    start_worker "cine_2ch_seg" ${PORT_SEG_2CH} "cine_2ch_seg_worker" 0 ${GPU_SEG_2CH}
    start_worker "cine_4ch_seg" ${PORT_SEG_4CH} "cine_4ch_seg_worker" 0 ${GPU_SEG_4CH}
    start_worker "cine_sa_seg"  ${PORT_SEG_SA}  "cine_sa_seg_worker"  0 ${GPU_SEG_SA}
    start_worker "lge_sa_seg"   ${PORT_SEG_LGE} "lge_sa_seg_worker"   0 ${GPU_SEG_LGE}

    start_worker "cds"   ${PORT_CDS}   "cds_worker"   0 ${GPU_CDS}
    start_worker "nicms" ${PORT_NICMS}  "nicms_worker" 0 ${GPU_NICMS}

    log_info "等待分割服务就绪后启动Metrics和MRG..."
    sleep 5
    start_metrics_worker
    start_mrg_worker
    start_mir_worker

    log_info "启动Sequence Analysis (依赖Agent服务)..."
    start_seq_worker

    log_info "所有Workers启动完成!"
}

# ============ Demo 服务 ============

start_demo_backend() {
    log_info "启动 Demo Backend (port ${DEMO_BACKEND_PORT}, env: ${CONDA_ENV_DEMO})..."

    if check_port $DEMO_BACKEND_PORT; then
        log_warn "端口 $DEMO_BACKEND_PORT 已被占用"
        local pid=$(lsof -Pi :$DEMO_BACKEND_PORT -sTCP:LISTEN -t 2>/dev/null)
        if [ -n "$pid" ]; then
            log_info "现有进程 PID: $pid"
            read -p "终止现有进程? [y/N] " confirm
            if [[ $confirm == [yY] ]]; then
                kill $pid 2>/dev/null
                log_info "等待端口释放..."
                sleep 3
            else
                log_error "无法启动 Demo Backend"
                return 1
            fi
        fi
    fi

    cd "${PROJECT_DIR}"

    nohup bash -c "
        source ${CONDA_PATH}/etc/profile.d/conda.sh
        conda activate ${CONDA_ENV_DEMO}
        cd ${PROJECT_DIR}
        python -m app.server --serve --port ${DEMO_BACKEND_PORT}
    " > "${LOG_DIR}/demo_backend.log" 2>&1 &

    local backend_pid=$!
    echo $backend_pid > "${FRONTEND_PID_DIR}/backend.pid"

    log_info "等待 Backend 启动（可能需要 20-30 秒）..."
    local max_wait=60
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if check_port $DEMO_BACKEND_PORT; then
            log_success "Demo Backend 已启动 (PID: $backend_pid, 耗时: ${waited}秒)"
            log_info "API Docs: http://localhost:${DEMO_BACKEND_PORT}/docs"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done

    log_error "Demo Backend 启动超时，查看日志: ${LOG_DIR}/demo_backend.log"
    return 1
}

start_demo_frontend() {
    log_info "启动 Demo Frontend (port ${DEMO_FRONTEND_PORT})..."

    if check_port $DEMO_FRONTEND_PORT; then
        log_warn "端口 $DEMO_FRONTEND_PORT 已被占用"
        local pid=$(lsof -Pi :$DEMO_FRONTEND_PORT -sTCP:LISTEN -t 2>/dev/null)
        if [ -n "$pid" ]; then
            log_info "现有进程 PID: $pid"
            read -p "终止现有进程? [y/N] " confirm
            if [[ $confirm == [yY] ]]; then
                kill $pid 2>/dev/null
                log_info "等待端口释放..."
                sleep 2
            else
                log_error "无法启动 Demo Frontend"
                return 1
            fi
        fi
    fi

    cd "${DEMO_DIR}"

    nohup python -m http.server $DEMO_FRONTEND_PORT > "${LOG_DIR}/demo_frontend.log" 2>&1 &
    local frontend_pid=$!
    echo $frontend_pid > "${FRONTEND_PID_DIR}/frontend.pid"

    log_info "等待 Frontend 启动..."
    local max_wait=5
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if check_port $DEMO_FRONTEND_PORT; then
            log_success "Demo Frontend 已启动 (PID: $frontend_pid)"
            log_info "Frontend URL: http://localhost:${DEMO_FRONTEND_PORT}"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done

    log_error "Demo Frontend 启动超时"
    return 1
}

start_demo() {
    log_info "启动 Demo 服务..."
    start_demo_backend
    if [ $? -eq 0 ]; then
        start_demo_frontend
    fi
    show_demo_info
}

# ============ 停止服务 ============

kill_tree() {
    local parent_pid=$1
    local signal=${2:-TERM}

    local children
    children=$(pgrep -P "$parent_pid" 2>/dev/null) || true
    for child in $children; do
        kill_tree "$child" "$signal"
    done

    kill -"$signal" "$parent_pid" 2>/dev/null || true
}

stop_all() {
    set +e

    log_info "停止所有服务..."

    stop_demo

    for pid_file in "${PID_DIR}"/*.pid; do
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            name=$(basename "$pid_file" .pid)
            if kill -0 "$pid" 2>/dev/null; then
                kill_tree "$pid" TERM
                sleep 1
                if kill -0 "$pid" 2>/dev/null; then
                    kill_tree "$pid" KILL
                    sleep 0.5
                fi
                log_info "已停止 ${name} (PID: ${pid}) 及其子进程"
            else
                log_warn "${name} 进程不存在 (PID: ${pid})"
            fi
            rm -f "$pid_file"
        fi
    done

    for port in ${PORT_CONTROLLER} ${PORT_AGENT} \
                ${PORT_SEG_2CH} ${PORT_SEG_4CH} ${PORT_SEG_SA} ${PORT_SEG_LGE} \
                ${PORT_CDS} ${PORT_NICMS} ${PORT_MRG} ${PORT_METRICS} ${PORT_MIR} ${PORT_SEQ}; do
        local port_pid
        port_pid=$(lsof -ti:${port} 2>/dev/null) || true
        if [ ! -z "$port_pid" ]; then
            local port_cmdline
            port_cmdline=$(tr '\0' ' ' < /proc/${port_pid}/cmdline 2>/dev/null) || true
            if echo "$port_cmdline" | grep -qF "${PROJECT_DIR}"; then
                log_warn "清理端口 ${port} 上的残留进程 ${port_pid}"
                kill -9 $port_pid 2>/dev/null || true
            else
                log_warn "端口 ${port} 被非本项目进程 ${port_pid} 占用，跳过"
            fi
        fi
    done

    log_info "按进程模式匹配清理残留Python进程（仅限本项目）..."
    local patterns=(
        "serve.controller"
        "serve.agent_worker"
        "serve.cine_2ch_seg_worker"
        "serve.cine_4ch_seg_worker"
        "serve.cine_sa_seg_worker"
        "serve.lge_sa_seg_worker"
        "serve.cds_worker"
        "serve.nicms_worker"
        "serve.metrics_worker"
        "serve.mrg_worker"
        "serve.rag_worker"
        "serve.seq_worker"
        "app.server"
    )
    for pattern in "${patterns[@]}"; do
        local pids=""
        local all_pids
        all_pids=$(pgrep -f "$pattern" 2>/dev/null) || true
        for p in $all_pids; do
            local cmdline
            cmdline=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null) || true
            if echo "$cmdline" | grep -qF "${PROJECT_DIR}"; then
                pids="$pids $p"
            fi
        done
        pids=$(echo "$pids" | xargs 2>/dev/null)
        if [ ! -z "$pids" ]; then
            log_warn "清理残留进程 (pattern: ${pattern}, 仅本项目): PIDs=${pids}"
            for p in $pids; do
                kill -9 "$p" 2>/dev/null || true
            done
        fi
    done

    sleep 2

    if command -v nvidia-smi &> /dev/null; then
        log_info "检查GPU上属于本项目的残留进程..."
        local gpu_pids
        gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ') || true
        if [ ! -z "$gpu_pids" ]; then
            local killed_any=false
            for gp in $gpu_pids; do
                if [ -z "$gp" ] || [ "$gp" = "pid" ]; then
                    continue
                fi
                local cmdline
                cmdline=$(tr '\0' ' ' < /proc/$gp/cmdline 2>/dev/null) || true
                if echo "$cmdline" | grep -qF "${PROJECT_DIR}"; then
                    local cmd_name
                    cmd_name=$(ps -p "$gp" -o comm= 2>/dev/null) || cmd_name="unknown"
                    log_warn "  杀死本项目GPU进程: PID=$gp ($cmd_name)"
                    kill -9 "$gp" 2>/dev/null || true
                    killed_any=true
                fi
            done
            if [ "$killed_any" = true ]; then
                sleep 3
            fi
        fi

        local remaining
        remaining=$(nvidia-smi --query-compute-apps=pid,name,used_gpu_memory --format=csv,noheader 2>/dev/null) || true
        if [ -z "$remaining" ]; then
            log_success "GPU显存已完全释放"
        else
            log_info "GPU上仍有其他进程（非本项目，未清理）:"
            echo "$remaining" | while IFS= read -r line; do
                if [ ! -z "$line" ]; then
                    log_info "  $line"
                fi
            done
        fi
    fi

    log_info "所有服务已停止"
}

stop_demo() {
    set +e

    log_info "停止 Demo 服务..."

    if [ -f "${FRONTEND_PID_DIR}/backend.pid" ]; then
        local pid=$(cat "${FRONTEND_PID_DIR}/backend.pid")
        if kill -0 $pid 2>/dev/null; then
            kill_tree $pid TERM
            sleep 1
            if kill -0 $pid 2>/dev/null; then
                kill_tree $pid KILL
            fi
            log_info "Demo Backend 已停止 (PID: $pid)"
        fi
        rm -f "${FRONTEND_PID_DIR}/backend.pid"
    fi

    if [ -f "${FRONTEND_PID_DIR}/frontend.pid" ]; then
        local pid=$(cat "${FRONTEND_PID_DIR}/frontend.pid")
        if kill -0 $pid 2>/dev/null; then
            kill_tree $pid TERM
            sleep 0.5
            if kill -0 $pid 2>/dev/null; then
                kill_tree $pid KILL
            fi
            log_info "Demo Frontend 已停止 (PID: $pid)"
        fi
        rm -f "${FRONTEND_PID_DIR}/frontend.pid"
    fi

    local stale_pids
    stale_pids=$(pgrep -f "app.server" 2>/dev/null) || true
    for p in $stale_pids; do
        local cmdline
        cmdline=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null) || true
        if echo "$cmdline" | grep -qF "${PROJECT_DIR}"; then
            kill -9 "$p" 2>/dev/null || true
        fi
    done
    stale_pids=$(pgrep -f "http.server ${DEMO_FRONTEND_PORT}" 2>/dev/null) || true
    for p in $stale_pids; do
        local cmdline
        cmdline=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null) || true
        if echo "$cmdline" | grep -qF "${DEMO_DIR}"; then
            kill -9 "$p" 2>/dev/null || true
        fi
    done

    for port in ${DEMO_BACKEND_PORT} ${DEMO_FRONTEND_PORT}; do
        local port_pid
        port_pid=$(lsof -ti:${port} 2>/dev/null) || true
        if [ ! -z "$port_pid" ]; then
            local port_cmdline
            port_cmdline=$(tr '\0' ' ' < /proc/${port_pid}/cmdline 2>/dev/null) || true
            if echo "$port_cmdline" | grep -qF "${PROJECT_DIR}"; then
                log_warn "清理端口 ${port} 上的残留进程 ${port_pid}"
                kill -9 $port_pid 2>/dev/null || true
            else
                log_warn "端口 ${port} 被非本项目进程 ${port_pid} 占用，跳过"
            fi
        fi
    done

    log_success "Demo 服务已停止"
}

# ============ 状态查询 ============

check_status() {
    echo ""
    echo "========== 服务状态 =========="
    echo ""

    if [ -f "${PID_DIR}/controller.pid" ]; then
        pid=$(cat "${PID_DIR}/controller.pid")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "Controller (${PORT_CONTROLLER}): ${GREEN}运行中${NC} (PID: ${pid})"
        else
            echo -e "Controller (${PORT_CONTROLLER}): ${RED}已停止${NC}"
        fi
    else
        echo -e "Controller (${PORT_CONTROLLER}): ${YELLOW}未启动${NC}"
    fi

    if [ -f "${PID_DIR}/agent_model.pid" ]; then
        pid=$(cat "${PID_DIR}/agent_model.pid")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "LLaVA Agent (${PORT_AGENT}): ${GREEN}运行中${NC} (PID: ${pid}) [GPU ${GPU_AGENT}]"
        else
            echo -e "LLaVA Agent (${PORT_AGENT}): ${RED}已停止${NC}"
        fi
    else
        echo -e "LLaVA Agent (${PORT_AGENT}): ${YELLOW}未启动${NC}"
    fi

    workers=(
        "cine_2ch_seg:${PORT_SEG_2CH}:${GPU_SEG_2CH}"
        "cine_4ch_seg:${PORT_SEG_4CH}:${GPU_SEG_4CH}"
        "cine_sa_seg:${PORT_SEG_SA}:${GPU_SEG_SA}"
        "lge_sa_seg:${PORT_SEG_LGE}:${GPU_SEG_LGE}"
        "cds:${PORT_CDS}:${GPU_CDS}"
        "nicms:${PORT_NICMS}:${GPU_NICMS}"
        "metrics:${PORT_METRICS}:CPU"
        "mrg:${PORT_MRG}:CPU"
        "mir:${PORT_MIR}:CPU"
        "seq:${PORT_SEQ}:CPU"
    )

    for w in "${workers[@]}"; do
        name=$(echo "$w" | cut -d: -f1)
        port=$(echo "$w" | cut -d: -f2)
        gpu=$(echo "$w" | cut -d: -f3)

        if [ -f "${PID_DIR}/${name}.pid" ]; then
            pid=$(cat "${PID_DIR}/${name}.pid")
            if kill -0 "$pid" 2>/dev/null; then
                echo -e "${name} (${port}): ${GREEN}运行中${NC} (PID: ${pid}) [GPU ${gpu}]"
            else
                echo -e "${name} (${port}): ${RED}已停止${NC}"
            fi
        else
            echo -e "${name} (${port}): ${YELLOW}未启动${NC}"
        fi
    done

    echo ""
    echo "========== Demo 状态 =========="
    echo ""

    if [ -f "${FRONTEND_PID_DIR}/backend.pid" ]; then
        pid=$(cat "${FRONTEND_PID_DIR}/backend.pid")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "Demo Backend (${DEMO_BACKEND_PORT}): ${GREEN}运行中${NC} (PID: ${pid})"
        else
            echo -e "Demo Backend (${DEMO_BACKEND_PORT}): ${RED}已停止${NC}"
        fi
    else
        echo -e "Demo Backend (${DEMO_BACKEND_PORT}): ${YELLOW}未启动${NC}"
    fi

    if [ -f "${FRONTEND_PID_DIR}/frontend.pid" ]; then
        pid=$(cat "${FRONTEND_PID_DIR}/frontend.pid")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "Demo Frontend (${DEMO_FRONTEND_PORT}): ${GREEN}运行中${NC} (PID: ${pid})"
        else
            echo -e "Demo Frontend (${DEMO_FRONTEND_PORT}): ${RED}已停止${NC}"
        fi
    else
        echo -e "Demo Frontend (${DEMO_FRONTEND_PORT}): ${YELLOW}未启动${NC}"
    fi

    echo ""
    echo "================================"
}

health_check() {
    echo ""
    echo "========== 健康检查 =========="
    echo ""

    if curl -s -X POST "http://localhost:${PORT_CONTROLLER}/list_models" > /dev/null 2>&1; then
        echo -e "Controller (${PORT_CONTROLLER}): ${GREEN}健康${NC}"
    else
        echo -e "Controller (${PORT_CONTROLLER}): ${RED}不可达${NC}"
    fi

    if curl -s "http://localhost:${PORT_AGENT}/health" > /dev/null 2>&1; then
        echo -e "LLaVA Agent (${PORT_AGENT}): ${GREEN}健康${NC} [GPU ${GPU_AGENT}]"
    else
        echo -e "LLaVA Agent (${PORT_AGENT}): ${RED}不可达${NC}"
    fi

    declare -A worker_info=(
        ["Cine2CHSeg"]="${PORT_SEG_2CH}:${GPU_SEG_2CH}"
        ["Cine4CHSeg"]="${PORT_SEG_4CH}:${GPU_SEG_4CH}"
        ["CineSASeg"]="${PORT_SEG_SA}:${GPU_SEG_SA}"
        ["LgeSASeg"]="${PORT_SEG_LGE}:${GPU_SEG_LGE}"
        ["CDS"]="${PORT_CDS}:${GPU_CDS}"
        ["NICMS"]="${PORT_NICMS}:${GPU_NICMS}"
        ["Metrics"]="${PORT_METRICS}:CPU"
        ["MRG"]="${PORT_MRG}:CPU"
        ["MIR"]="${PORT_MIR}:CPU"
        ["Seq"]="${PORT_SEQ}:CPU"
    )

    for name in "${!worker_info[@]}"; do
        info="${worker_info[$name]}"
        port="${info%%:*}"
        gpu="${info##*:}"
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo -e "${name} (${port}): ${GREEN}健康${NC} [GPU ${gpu}]"
        else
            echo -e "${name} (${port}): ${RED}不可达${NC}"
        fi
    done

    echo ""
    echo "========== Demo 健康检查 =========="
    echo ""

    if curl -s "http://localhost:${DEMO_BACKEND_PORT}/health" > /dev/null 2>&1; then
        echo -e "Demo Backend (${DEMO_BACKEND_PORT}): ${GREEN}健康${NC}"
    else
        echo -e "Demo Backend (${DEMO_BACKEND_PORT}): ${RED}不可达${NC}"
    fi

    if curl -s "http://localhost:${DEMO_FRONTEND_PORT}" > /dev/null 2>&1; then
        echo -e "Demo Frontend (${DEMO_FRONTEND_PORT}): ${GREEN}健康${NC}"
    else
        echo -e "Demo Frontend (${DEMO_FRONTEND_PORT}): ${RED}不可达${NC}"
    fi

    echo ""
}

show_demo_info() {
    echo ""
    echo -e "${GREEN}═══════════════════ Demo Access URLs ═══════════════════${NC}"
    echo ""
    echo -e "  ${YELLOW}Frontend:${NC}     http://localhost:${DEMO_FRONTEND_PORT}"
    echo -e "  ${YELLOW}API Docs:${NC}     http://localhost:${DEMO_BACKEND_PORT}/docs"
    echo -e "  ${YELLOW}Health Check:${NC} http://localhost:${DEMO_BACKEND_PORT}/health"
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
    echo ""
}

show_gpu_config() {
    echo ""
    echo "========== 环境与GPU配置 =========="
    echo ""
    echo "Project Root: ${PROJECT_DIR}"
    echo "Conda路径: ${CONDA_PATH}"
    echo ""
    echo "Conda环境:"
    echo "  Agent模型:    ${CONDA_ENV_AGENT}"
    echo "  Expert模型:   ${CONDA_ENV_EXPERT}"
    echo "  Demo前后端:   ${CONDA_ENV_DEMO}"
    echo ""
    echo "GPU配置:"
    echo "  Agent模型:    GPU ${GPU_AGENT}"
    echo ""
    echo "  分割模型 (Segmentation):"
    echo "    Cine2CHSeg:  GPU ${GPU_SEG_2CH}"
    echo "    Cine4CHSeg:  GPU ${GPU_SEG_4CH}"
    echo "    CineSASeg:   GPU ${GPU_SEG_SA}"
    echo "    LgeSASeg:    GPU ${GPU_SEG_LGE}"
    echo ""
    echo "  分类模型 (Classification):"
    echo "    CDS:   GPU ${GPU_CDS}"
    echo "    NICMS: GPU ${GPU_NICMS}"
    echo ""
    echo "  其他服务:"
    echo "    Metrics (Cardiac Metrics Calc):   CPU"
    echo "    MRG (Medical Report Generation): CPU (orchestrator)"
    echo "    MIR (Medical Info Retrieval):     CPU"
    echo "    Seq (Sequence Analysis):           CPU"
    echo ""
    echo "Demo配置:"
    echo "  Backend:  python -m app.server --serve --port ${DEMO_BACKEND_PORT}"
    echo "  Frontend: python -m http.server ${DEMO_FRONTEND_PORT}  (from ${DEMO_DIR})"
    echo ""
    echo "修改配置: 编辑此脚本顶部的变量"
    echo "===================================="
}

show_help() {
    echo "心脏医学影像Agent系统 - 服务管理脚本"
    echo ""
    echo "使用方法 (从 MMedAgent 根目录运行):"
    echo "  ./app/start.sh all              启动所有服务 (Controller + Agent + Workers)"
    echo "  ./app/start.sh controller       只启动Controller"
    echo "  ./app/start.sh agent            只启动LLaVA Agent模型"
    echo "  ./app/start.sh workers          只启动所有Expert Workers"
    echo "  ./app/start.sh demo             只启动Demo (Backend + Frontend)"
    echo "  ./app/start.sh frontend         只启动前端 (Frontend only)"
    echo "  ./app/start.sh full             启动完整系统 (所有服务 + Demo)"
    echo "  ./app/start.sh stop [service]   停止所有服务或指定服务"
    echo "  ./app/start.sh stop-demo        只停止Demo服务"
    echo "  ./app/start.sh status           查看服务状态"
    echo "  ./app/start.sh health           健康检查 (通过HTTP)"
    echo "  ./app/start.sh restart          重启所有服务"
    echo "  ./app/start.sh gpu              显示GPU配置"
    echo ""
    echo "服务架构:"
    echo "  Controller (${PORT_CONTROLLER}) -> LLaVA Agent (${PORT_AGENT}) -> Expert Workers"
    echo ""
    echo "日志目录: ${LOG_DIR}"
    echo "PID目录: ${PID_DIR}"
}

# ============ 主入口 ============

print_banner

case "$1" in
    all)
        start_controller
        sleep 2
        start_agent
        sleep 30
        start_workers
        sleep 30
        health_check
        ;;
    controller)
        start_controller
        ;;
    agent)
        start_agent
        ;;
    workers)
        start_workers
        sleep 30
        health_check
        ;;
    demo)
        start_demo
        ;;
    frontend)
        start_demo_frontend
        show_demo_info
        ;;
    full)
        log_info "启动完整系统 (所有服务 + Demo)..."
        start_controller
        sleep 2
        start_agent
        sleep 30
        start_workers
        sleep 30
        start_demo
        health_check
        ;;
    stop)
        stop_all
        ;;
    stop-demo)
        stop_demo
        ;;
    status)
        check_status
        ;;
    health)
        health_check
        ;;
    gpu)
        show_gpu_config
        ;;
    restart)
        stop_all
        sleep 2
        start_controller
        sleep 2
        start_agent
        sleep 30
        start_workers
        sleep 30
        health_check
        ;;
    *)
        show_help
        ;;
esac
