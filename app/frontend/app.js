// 配置
const API_BASE_URL = 'http://localhost:8005';

// 状态
let selectedFiles = [];
let selectedTask = 'mr';
let isProcessing = false;
let currentSessionId = null;

// DOM元素
const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');
const clearAllBtn = document.getElementById('clearAllBtn');
const uploadZone = document.getElementById('uploadZone');
const processingBar = document.getElementById('processingBar');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const taskDescription = document.getElementById('taskDescription');

const taskDescriptions = {
  mr: 'Cardiac MR mode: Upload cardiac MRI files (.zip/.nii.gz) → Agent identifies sequences → Smart frame extraction → Agent decides API → Expert model → Download results (seg labels, NIFTI, PDF reports).',
  ct: 'Cardiac CT mode: Coming soon.',
  us: 'Cardiac Ultrasound mode: Coming soon.',
  ecg: 'ECG analysis mode: Coming soon.',
};

// 初始化
document.addEventListener('DOMContentLoaded', async () => {
  document.getElementById('welcomeTime').textContent = formatTime(new Date());
  await checkServerStatus();
  setupEventListeners();
  // 自动创建会话
  await createNewSession();
});

// 设置事件监听
function setupEventListeners() {
  // 任务选择
  document.querySelectorAll('.task-btn:not(.disabled)').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.task-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      selectedTask = btn.dataset.task;
      taskDescription.textContent = taskDescriptions[selectedTask];
    });
  });

  // 文件上传
  fileInput.addEventListener('change', handleFileSelect);
  
  // 拖拽上传
  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
  });
  
  uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
  });
  
  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    const files = Array.from(e.dataTransfer.files);
    addFiles(files);
  });

  // 清除文件
  clearAllBtn.addEventListener('click', clearFiles);

  // 输入监听
  messageInput.addEventListener('input', () => {
    messageInput.style.height = 'auto';
    messageInput.style.height = messageInput.scrollHeight + 'px';
    updateSendButton();
  });

  messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // 发送按钮
  sendBtn.addEventListener('click', sendMessage);

  // 快速提示
  document.querySelectorAll('.quick-prompt[data-prompt]').forEach(btn => {
    btn.addEventListener('click', () => {
      messageInput.value = btn.dataset.prompt;
      messageInput.dispatchEvent(new Event('input'));
    });
  });
}

// 检查服务器状态
async function checkServerStatus() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, { 
      method: 'GET',
      timeout: 5000 
    });
    if (response.ok) {
      statusDot.style.background = 'var(--success)';
      statusText.textContent = 'Online';
    } else {
      throw new Error('Service unavailable');
    }
  } catch (error) {
    statusDot.style.background = 'var(--error)';
    statusText.textContent = 'Offline';
  }
}

// 文件处理
function handleFileSelect(e) {
  const files = Array.from(e.target.files);
  addFiles(files);
  e.target.value = '';
}

function addFiles(files) {
  files.forEach(file => {
    // 检查文件类型 (V15增强：支持PNG/JPG图像)
    const validExtensions = ['.zip', '.nii', '.nii.gz', '.png', '.jpg', '.jpeg', '.gif', '.webp'];
    const ext = '.' + file.name.split('.').slice(-2).join('.').toLowerCase();
    const ext2 = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!validExtensions.some(e => ext.includes(e) || ext2.includes(e))) {
      alert(`Unsupported file type: ${file.name}\nPlease upload .zip (DCM folder archive), .nii.gz file, or PNG/JPG image`);
      return;
    }
    
    selectedFiles.push(file);
  });
  
  updateFileList();
  updateSendButton();
}

function updateFileList() {
  fileList.innerHTML = '';
  
  selectedFiles.forEach((file, index) => {
    const item = document.createElement('div');
    item.className = 'file-item';
    item.innerHTML = `
      <span class="file-icon">${getFileIcon(file.name)}</span>
      <div class="file-info">
        <div class="file-name">${file.name}</div>
        <div class="file-size">${formatFileSize(file.size)}</div>
      </div>
      <button class="remove-btn" onclick="removeFile(${index})">✕</button>
    `;
    fileList.appendChild(item);
  });
  
  clearAllBtn.style.display = selectedFiles.length > 0 ? 'block' : 'none';
}

function removeFile(index) {
  selectedFiles.splice(index, 1);
  updateFileList();
  updateSendButton();
}

function clearFiles() {
  selectedFiles = [];
  updateFileList();
  updateSendButton();
}

function getFileIcon(filename) {
  const lowerName = filename.toLowerCase();
  if (lowerName.endsWith('.zip')) {
    return '📦';
  } else if (lowerName.endsWith('.nii.gz') || lowerName.endsWith('.nii')) {
    return '🏥';
  } else if (lowerName.endsWith('.dcm')) {
    return '📷';
  } else if (/\.(png|jpg|jpeg)$/i.test(filename)) {
    return '🖼️';
  }
  return '📄';
}

// 发送消息
function updateSendButton() {
  sendBtn.disabled = isProcessing || (messageInput.value.trim() === '' && selectedFiles.length === 0);
}

async function sendMessage() {
  const message = messageInput.value.trim();
  
  if (isProcessing || (message === '' && selectedFiles.length === 0)) {
    return;
  }

  // 添加用户消息
  addUserMessage(message, selectedFiles);
  
  // 清空输入
  messageInput.value = '';
  messageInput.style.height = 'auto';
  const currentFiles = [...selectedFiles];
  clearFiles();
  
  // 显示处理状态
  isProcessing = true;
  updateSendButton();
  processingBar.classList.add('active');
  const typingId = addTypingIndicator();

  try {
    // 准备请求
    const formData = new FormData();
    formData.append('message', message);
    formData.append('model', 'agent');
    
    // 添加会话ID
    if (currentSessionId) {
      formData.append('session_id', currentSessionId);
    }
    
    formData.append('task_type', selectedTask);
    
    // 添加文件
    currentFiles.forEach(file => {
      formData.append('files', file, file.name);
    });

    // 发送请求
    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`Request failed: ${response.status}`);
    }

    const data = await response.json();
    
    // 更新会话ID
    if (data.session_id) {
      currentSessionId = data.session_id;
      document.getElementById('currentSessionId').textContent = currentSessionId;
    }
    
    // 移除加载指示器
    removeTypingIndicator(typingId);
    
    // 添加机器人回复
    addBotMessage(data);

  } catch (error) {
    removeTypingIndicator(typingId);
    addBotMessage({
      response: `Sorry, error processing request: ${error.message}`,
      error: true
    });
  } finally {
    isProcessing = false;
    updateSendButton();
    processingBar.classList.remove('active');
  }
}

// 添加消息到聊天区
function addUserMessage(text, files) {
  const messageDiv = document.createElement('div');
  messageDiv.className = 'message user';
  
  let content = `<div class="message-content">${escapeHtml(text)}`;
  
  if (files.length > 0) {
    content += '<div class="message-files">';
    files.forEach(file => {
      content += `<span class="message-file">${getFileIcon(file.name)} ${file.name}</span>`;
    });
    content += '</div>';
  }
  
  content += '</div>';
  content += `<div class="message-meta"><span>${formatTime(new Date())}</span></div>`;
  
  messageDiv.innerHTML = content;
  chatMessages.appendChild(messageDiv);
  scrollToBottom();
}

function addBotMessage(data) {
  const messageDiv = document.createElement('div');
  messageDiv.className = 'message bot wide';
  
  let content = '<div class="message-content">';
  
  // 判断是否需要结构化对话可视化
  const hasApiName = data.api_name && data.api_name !== 'Agent VQA';
  const hasFirstResponse = data.first_response;
  const hasTwoTurns = hasApiName && hasFirstResponse;
  
  if (hasTwoTurns) {
    // ========== 两轮对话结构化可视化（有外部API调用） ==========
    content += '<div class="conversation-flow">';
    
    const parsedFirst = parseAgentResponse(data.first_response);
    
    // Step 1: thoughts
    if (parsedFirst.thoughts) {
      content += `
        <div class="flow-step thinking">
          <div class="flow-step-header">🧠 Agent Analysis</div>
          <div class="flow-step-content">${escapeHtml(parsedFirst.thoughts)}</div>
        </div>
      `;
    }
    
    // Step 2: actions → API
    if (data.api_name) {
      content += `
        <div class="flow-step action">
          <div class="flow-step-header">🚀 API Decision</div>
          <div class="flow-step-content">
            <span class="api-tag">API: ${data.api_name}</span>
          </div>
        </div>
      `;
    }
    
    // Separator
    content += '<div class="turn-separator">Expert Model Execution</div>';
    
    // Step 3: value (conclusion from Turn 2)
    const parsedFinal = parseAgentResponse(data.response);
    const finalOutput = parsedFinal.value || parsedFinal.raw || data.response || 'Processing complete';
    const isRag = data.api_name === 'Medical Info Retrieval';
    const renderedConclusion = isRag ? renderMarkdown(finalOutput) : formatResponse(finalOutput);
    content += `
      <div class="flow-step conclusion">
        <div class="flow-step-header">${isRag ? '📖 Medical Knowledge' : '📊 Agent Conclusion'}</div>
        <div class="flow-step-content">${renderedConclusion}</div>
      </div>
    `;
    
    content += '</div>';
  } else if (data.api_name === 'Agent VQA' || !hasApiName) {
    // ========== V26: Agent VQA — 与API同构的3步可视化 ==========
    // 优先从 first_response（完整Agent原始输出）解析 thoughts/actions/value
    const fullAgentText = data.first_response || data.response;
    const parsedResp = parseAgentResponse(fullAgentText);
    
    content += '<div class="conversation-flow">';
    
    // Step 1: thoughts🤔 — Agent Analysis
    if (parsedResp.thoughts) {
      content += `
        <div class="flow-step thinking">
          <div class="flow-step-header">🧠 Agent Analysis</div>
          <div class="flow-step-content">${escapeHtml(parsedResp.thoughts)}</div>
        </div>
      `;
    }
    
    // Step 2: actions🚀 → Agent VQA（与API展示风格一致）
    content += `
      <div class="flow-step action">
        <div class="flow-step-header">🚀 API Decision</div>
        <div class="flow-step-content">
          <span class="api-tag">Agent VQA</span>
        </div>
      </div>
    `;
    
    // Step 3: value👉 → VQA 优化展示
    const vqaValue = parsedResp.value || data.response || 'Processing complete';
    content += `
      <div class="flow-step conclusion">
        <div class="flow-step-header">💡 Answer</div>
        <div class="flow-step-content">${renderVQAContent(vqaValue)}</div>
      </div>
    `;
    
    content += '</div>';
  } else {
    // Fallback: plain response
    content += formatResponse(data.response || 'Processing complete');
    if (data.api_name) {
      content += `<div class="api-tag">API: ${data.api_name}</div>`;
    }
  }
  
  // Prediction result (standalone classification without report)
  const hasReportClassification = data.report_data && data.report_data.sections &&
    data.report_data.sections.some(s => s.name === 'Classification Results');
  if (data.prediction && !hasReportClassification) {
    content += `
      <div class="result-box">
        <div class="result-label">Classification Result</div>
        <div class="result-value" style="color: #fbbf24;">${data.prediction}</div>
      </div>
    `;
  }
  
  // 检测到的序列
  if (data.detected_sequences && data.detected_sequences.length > 0) {
    content += '<div class="sequence-results" style="margin-top: 8px;">';
    content += '<span style="font-size: 11px; color: var(--text-secondary); margin-right: 6px;">🔍 Detected:</span>';
    data.detected_sequences.forEach((seq, idx) => {
      const label = typeof seq === 'string' ? seq.replace(/_/g, ' ').toUpperCase() : (seq.modality || seq);
      content += `<span class="sequence-tag"><span class="modality">${label}</span></span>`;
    });
    content += '</div>';
  }
  
  // 显示上传的PNG图像
  if (data.image_urls && data.image_urls.length > 0) {
    content += `
      <div class="frame-preview" style="border-left: 3px solid rgba(78, 205, 196, 0.45);">
        <div class="frame-preview-title">🖼️ Uploaded Images (${data.image_urls.length} images)</div>
        <div class="frame-grid">
    `;
    data.image_urls.forEach((img, index) => {
      const imgUrl = API_BASE_URL + img.url;
      content += `
        <div class="frame-item orig-frame" onclick="openImageModal('${imgUrl}', '${img.filename}', ${index})">
          <img src="${imgUrl}" alt="${img.filename}" loading="lazy">
          <div class="frame-label">${img.filename}</div>
        </div>
      `;
    });
    content += '</div></div>';
  }
  
  // Frame extraction preview
  if (data.frame_urls && data.frame_urls.length > 0) {
    content += `
      <div class="frame-preview" style="border-left: 3px solid rgba(78, 205, 196, 0.45);">
        <div class="frame-preview-title">🖼️ Extracted Frames (${data.frame_urls.length} frames)</div>
        <div class="frame-grid">
    `;
    data.frame_urls.forEach((frame, index) => {
      const imgUrl = API_BASE_URL + frame.url;
      content += `
        <div class="frame-item orig-frame" onclick="openImageModal('${imgUrl}', '${frame.filename}', ${frame.frame_index})">
          <img src="${imgUrl}" alt="Frame ${frame.frame_index}" loading="lazy">
          <div class="frame-label">Frame #${frame.frame_index}</div>
        </div>
      `;
    });
    content += '</div></div>';
  }
  
  // Segmentation results display (multi-frame)
  if (data.seg_result && data.seg_result.seg_files && data.seg_result.seg_files.length > 0) {
    content += `
      <div class="frame-preview" style="border-left: 3px solid rgba(244, 114, 182, 0.5);">
        <div class="frame-preview-title">✂️ Segmentation Results (${data.seg_result.seg_files.length} frames)</div>
        <div class="frame-grid">
    `;
    data.seg_result.seg_files.forEach((seg, index) => {
      const segImgUrl = API_BASE_URL + seg.url;
      content += `
        <div class="frame-item seg-result" onclick="openImageModal('${segImgUrl}', '${seg.filename}', ${seg.frame_index})">
          <img src="${segImgUrl}" alt="Seg Frame ${seg.frame_index}" loading="lazy">
          <div class="frame-label">Frame #${seg.frame_index}</div>
        </div>
      `;
    });
    content += '</div></div>';
  }
  
  // 医学报告生成结果显示
  if (data.report_data && data.report_data.sections && data.report_data.sections.length > 0) {
    const reportId = 'report-' + Date.now();
    content += `
      <div class="medical-report">
        <div class="report-header">
          <span class="icon">📋</span>
          <h3>${data.report_data.title || 'Cardiac Function Evaluation Report'}</h3>
        </div>
    `;
    
    // V23: Highlight 关键指标区域（6个Bland-Altman关键指标）
    const highlightDefs = [
      { key: 'LV_EF',  abbr: 'LVEF',  name: 'LV Ejection Fraction' },
      { key: 'LV_EDV', abbr: 'LVEDV', name: 'LV End-Diastolic Volume' },
      { key: 'LV_ESV', abbr: 'LVESV', name: 'LV End-Systolic Volume' },
      { key: 'LV_SV',  abbr: 'SV',    name: 'Stroke Volume' },
      { key: 'LV_Mass',abbr: 'LVM',   name: 'LV Myocardial Mass' },
      { key: 'LV_LD',  abbr: 'LVEDD', name: 'LV End-Diastolic Diameter' },
    ];
    
    // Collect highlight items from all sections
    const highlightItems = [];
    if (data.metrics || data.report_data.sections) {
      highlightDefs.forEach(def => {
        // Try metrics first
        let value = data.metrics ? data.metrics[def.key] : undefined;
        let unit = '';
        let normalRange = '';
        let status = 'normal';
        // Also search sections for more info
        data.report_data.sections.forEach(section => {
          section.items.forEach(item => {
            if (item.key === def.key) {
              if (value === undefined) value = item.value;
              unit = item.unit;
              normalRange = item.normal_range;
              status = item.status || 'normal';
            }
          });
        });
        if (value !== undefined && value !== null) {
          highlightItems.push({ ...def, value, unit, normalRange, status });
        }
      });
    }
    
    if (highlightItems.length > 0) {
      content += `
        <div class="report-section">
          <div class="report-section-toggle" onclick="toggleReportSection(this)">
            <span class="toggle-arrow">▼</span>
            <span>⭐</span>
            <span>Key Clinical Metrics (Highlight)</span>
          </div>
          <div class="report-section-body">
            <div class="highlight-section">
              <div class="highlight-grid">
      `;
      highlightItems.forEach(hl => {
        const statusLabel = getStatusLabel(hl.status);
        content += `
          <div class="highlight-card">
            <div class="hl-abbr">${hl.abbr}</div>
            <div class="hl-name">${hl.name}</div>
            <div>
              <span class="hl-value">${formatMetricValue(hl.value)}</span>
              <span class="hl-unit">${hl.unit}</span>
              ${statusLabel ? `<span class="hl-status ${hl.status}">${statusLabel}</span>` : ''}
            </div>
            ${hl.normalRange ? `<div class="hl-range">Normal: ${hl.normalRange}</div>` : ''}
          </div>
        `;
      });
      content += '</div></div></div></div>';
    }
    
    // V23: 显示各部分指标（每个 section 可折叠）
    data.report_data.sections.forEach((section, sIdx) => {
      const sectionBodyId = `${reportId}-sec-${sIdx}`;
      content += `
        <div class="report-section">
          <div class="report-section-toggle" onclick="toggleReportSection(this)">
            <span class="toggle-arrow">▼</span>
            <span>${getSectionIcon(section.name)}</span>
            <span>${section.name}</span>
          </div>
          <div class="report-section-body" id="${sectionBodyId}">
            <div class="report-metrics${section.name === 'Classification Results' ? ' classification-metrics' : ''}">
      `;
      
      section.items.forEach(item => {
        const statusClass = item.status || 'normal';
        const statusLabel = getStatusLabel(item.status);
        // Check if this is a highlighted metric
        const hlDef = highlightDefs.find(d => d.key === item.key);
        const isHighlight = !!hlDef;
        content += `
          <div class="metric-item" ${isHighlight ? 'style="border-left: 3px solid #fbbf24;"' : ''}>
            <div class="metric-name">${item.name} ${isHighlight ? `<span style="font-size:9px; color:#fbbf24; font-weight:700; margin-left:4px;">${hlDef.abbr}</span>` : ''}</div>
            <div class="metric-value-row">
              <span class="metric-value">${formatMetricValue(item.value)}</span>
              <span class="metric-unit">${item.unit}</span>
              ${statusLabel ? `<span class="metric-status ${statusClass}">${statusLabel}</span>` : ''}
            </div>
            <div class="metric-range">Normal: ${item.normal_range}</div>
          </div>
        `;
      });
      
      content += '</div></div></div>';
    });
    
    content += '</div>';
  }
  // 如果有 metrics 但没有 report_data，显示简化版本
  else if (data.metrics && Object.keys(data.metrics).length > 0) {
    content += `
      <div class="medical-report">
        <div class="report-header">
          <span class="icon">📋</span>
          <h3>Cardiac Metrics</h3>
        </div>
        <div class="report-metrics">
    `;
    
    const keyMetrics = ['LV_EF', 'RV_EF', 'LV_EDV', 'LV_ESV', 'LV_SV', 'LV_Mass', 'RV_EDV', 'RV_ESV'];
    keyMetrics.forEach(key => {
      const value = data.metrics[key];
      if (value !== undefined && value !== null) {
        const unit = key.includes('EF') ? '%' : (key.includes('Mass') ? 'g' : 'ml');
        content += `
          <div class="metric-item">
            <div class="metric-name">${key.replace(/_/g, ' ')}</div>
            <div class="metric-value-row">
              <span class="metric-value">${formatMetricValue(value)}</span>
              <span class="metric-unit">${unit}</span>
            </div>
          </div>
        `;
      }
    });
    
    content += '</div></div>';
  }
  
  // ========== 下载按钮区域 ==========
  if (data.download_urls && data.download_urls.length > 0) {
    content += `
      <div class="download-section">
        <div class="download-section-title">📥 Downloadable Files</div>
        <div class="download-grid">
    `;
    data.download_urls.forEach(dl => {
      const typeClass = `type-${dl.type}`;
      const iconMap = {
        'nifti': '🏥',
        'seg_label': '🏷️',
        'report_pdf': '📄',
      };
      const icon = iconMap[dl.type] || '📁';
      const downloadUrl = API_BASE_URL + dl.url;
      content += `
        <a class="download-btn ${typeClass}" href="${downloadUrl}" download="${dl.filename}" target="_blank">
          <span class="dl-icon">${icon}</span>
          <span>
            <span class="dl-label">${dl.label}</span><br>
            <span class="dl-filename">${dl.filename}</span>
          </span>
        </a>
      `;
    });
    content += '</div></div>';
  }
  
  // 错误状态
  if (data.error) {
    content = content.replace('result-value success', 'result-value error');
  }
  
  content += '</div>';
  content += `<div class="message-meta"><span>Cardiac Agent</span><span>•</span><span>${formatTime(new Date())}</span></div>`;
  
  messageDiv.innerHTML = content;
  chatMessages.appendChild(messageDiv);
  scrollToBottom();
}

function addTypingIndicator() {
  const id = 'typing-' + Date.now();
  const messageDiv = document.createElement('div');
  messageDiv.className = 'message bot';
  messageDiv.id = id;
  messageDiv.innerHTML = `
    <div class="message-content">
      <div class="typing-indicator">
        <span></span><span></span><span></span>
      </div>
      <span style="margin-left: 10px; color: var(--text-muted); font-size: 12px;">Processing...</span>
    </div>
  `;
  chatMessages.appendChild(messageDiv);
  scrollToBottom();
  return id;
}

function removeTypingIndicator(id) {
  const element = document.getElementById(id);
  if (element) {
    element.remove();
  }
}

function clearChat() {
  const welcomeMsg = chatMessages.firstElementChild;
  chatMessages.innerHTML = '';
  chatMessages.appendChild(welcomeMsg);
}

// 工具函数
function formatTime(date) {
  return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

function formatFileSize(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function formatResponse(text) {
  // Format response text
  let formatted = escapeHtml(text);
  
  // Process special markers (clean up for plain display)
  formatted = formatted.replace(/"thoughts🤔"/g, '');
  formatted = formatted.replace(/"actions🚀"/g, '');
  formatted = formatted.replace(/"value👉"/g, '');
  
  // Line break handling
  formatted = formatted.replace(/\n/g, '<br>');
  
  return formatted;
}

/**
 * 轻量 Markdown → HTML 渲染器
 * 支持: **bold**, *italic*, ### headers, 有序/无序列表, 嵌套子列表, 链接, Note 段落
 */
function renderMarkdown(raw) {
  if (!raw) return '';
  // 清理 Agent 结构化标记
  let text = raw.replace(/"thoughts🤔"[\s\S]*?"actions🚀"[\s\S]*?"value👉"\s*/g, '');
  text = text.replace(/"thoughts🤔"/g, '').replace(/"actions🚀"/g, '').replace(/"value👉"/g, '');
  // 清理反馈 prompt 残留（"Answer my first question: ..." 及其后所有内容）
  text = text.replace(/\n*Answer my first question:[\s\S]*$/i, '');
  text = text.trim();
  if (!text) return '';

  const lines = text.split('\n');
  let html = '';
  let inOl = false, inUl = false, inSubUl = false;

  function closeLists() {
    let s = '';
    if (inSubUl) { s += '</ul></li>'; inSubUl = false; }
    if (inUl)    { s += '</ul>'; inUl = false; }
    if (inOl)    { s += '</ol>'; inOl = false; }
    return s;
  }
  function closeSubList() {
    let s = '';
    if (inSubUl) { s += '</ul></li>'; inSubUl = false; }
    return s;
  }

  function inlineFormat(t) {
    t = escapeHtml(t);
    // links: (url)
    t = t.replace(/\((https?:\/\/[^\s)]+)\)/g, '(<a href="$1" target="_blank" rel="noopener">link</a>)');
    // bold
    t = t.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    // italic
    t = t.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>');
    return t;
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trimEnd();

    // blank line
    if (trimmed === '') {
      html += closeLists();
      continue;
    }

    // headers
    const hMatch = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (hMatch) {
      html += closeLists();
      const level = Math.min(hMatch[1].length + 2, 6); // # → h3, ## → h4, etc.
      html += `<h${level}>${inlineFormat(hMatch[2])}</h${level}>`;
      continue;
    }

    // Note/Source line — 提取 URL 作为可点击的参考链接
    if (/^note:/i.test(trimmed)) {
      html += closeLists();
      const noteUrlMatch = trimmed.match(/\((https?:\/\/[^\s)]+)\)/);
      if (noteUrlMatch) {
        const noteUrl = noteUrlMatch[1];
        const noteText = trimmed.replace(/\s*\(https?:\/\/[^\s)]+\)\s*$/, '');
        html += `<div class="md-source">${inlineFormat(noteText)}<br><a href="${escapeHtml(noteUrl)}" target="_blank" rel="noopener">📎 ${escapeHtml(noteUrl)}</a></div>`;
      } else {
        html += `<div class="md-source">${inlineFormat(trimmed)}</div>`;
      }
      continue;
    }

    // sub-bullet:   * item  or   - item (3+ leading spaces)
    const subBullet = trimmed.match(/^(\s{2,})[*\-]\s+(.+)$/);
    if (subBullet && (inOl || inUl)) {
      if (!inSubUl) {
        // 回退上一个 </li>，嵌套 <ul>
        html = html.replace(/<\/li>$/, '');
        html += '<ul>';
        inSubUl = true;
      }
      html += `<li>${inlineFormat(subBullet[2])}</li>`;
      continue;
    }

    // ordered list: 1. item
    const olMatch = trimmed.match(/^(\d+)\.\s+(.+)$/);
    if (olMatch) {
      html += closeSubList();
      if (!inOl) {
        html += closeLists();
        html += '<ol>';
        inOl = true;
      }
      html += `<li>${inlineFormat(olMatch[2])}</li>`;
      continue;
    }

    // unordered list: * item  or - item (top-level)
    const ulMatch = trimmed.match(/^[*\-]\s+(.+)$/);
    if (ulMatch) {
      html += closeSubList();
      if (!inUl) {
        html += closeLists();
        html += '<ul>';
        inUl = true;
      }
      html += `<li>${inlineFormat(ulMatch[1])}</li>`;
      continue;
    }

    // Markdown table: rows starting and ending with |
    if (/^\|.+\|$/.test(trimmed)) {
      html += closeLists();
      const tableRows = [trimmed];
      while (i + 1 < lines.length && /^\|.+\|$/.test(lines[i + 1].trimEnd())) {
        i++;
        tableRows.push(lines[i].trimEnd());
      }
      html += renderTable(tableRows);
      continue;
    }

    // plain paragraph
    html += closeLists();
    html += `<p>${inlineFormat(trimmed)}</p>`;
  }
  html += closeLists();

  return `<div class="md-rendered">${html}</div>`;

  function renderTable(rows) {
    if (rows.length < 2) return rows.map(r => `<p>${inlineFormat(r)}</p>`).join('');

    function splitCells(row) {
      return row.replace(/^\|/, '').replace(/\|$/, '').split('|').map(c => c.trim());
    }

    const isSep = /^\|[\s:]*-{2,}[\s:]*(\|[\s:]*-{2,}[\s:]*)*\|$/.test(rows[1]);
    const headers = isSep ? splitCells(rows[0]) : null;
    const dataStart = isSep ? 2 : 0;

    // Parse alignment from separator row
    let aligns = [];
    if (isSep) {
      splitCells(rows[1]).forEach(sep => {
        const left = sep.startsWith(':');
        const right = sep.endsWith(':');
        aligns.push(left && right ? 'center' : right ? 'right' : 'left');
      });
    }

    function formatCell(text) {
      let c = inlineFormat(text);
      c = c.replace(/&lt;br\s*\/?&gt;/gi, '<br>');
      return c;
    }

    let t = '<div class="table-wrap"><table>';
    if (headers) {
      t += '<thead><tr>';
      headers.forEach((h, idx) => {
        const align = aligns[idx] ? ` style="text-align:${aligns[idx]}"` : '';
        t += `<th${align}>${formatCell(h)}</th>`;
      });
      t += '</tr></thead>';
    }
    t += '<tbody>';
    for (let r = dataStart; r < rows.length; r++) {
      const cells = splitCells(rows[r]);
      t += '<tr>';
      cells.forEach((c, idx) => {
        const align = aligns[idx] ? ` style="text-align:${aligns[idx]}"` : '';
        t += `<td${align}>${formatCell(c)}</td>`;
      });
      t += '</tr>';
    }
    t += '</tbody></table></div>';
    return t;
  }
}

// 解析Agent响应中的thoughts/actions/value结构
function parseAgentResponse(text) {
  if (!text) return { thoughts: null, actions: null, value: null, raw: text };
  
  let thoughts = null;
  let actions = null;
  let value = null;
  
  // 尝试解析 "thoughts🤔" ... "actions🚀" ... "value👉" ... 格式
  const thoughtsMatch = text.match(/"thoughts🤔"\s*([\s\S]*?)(?="actions🚀"|"value👉"|$)/);
  const actionsMatch = text.match(/"actions🚀"\s*([\s\S]*?)(?="value👉"|$)/);
  const valueMatch = text.match(/"value👉"\s*([\s\S]*?)$/);
  
  if (thoughtsMatch) thoughts = thoughtsMatch[1].trim();
  if (actionsMatch) actions = actionsMatch[1].trim();
  if (valueMatch) value = valueMatch[1].trim();
  
  // 清理引号
  if (thoughts) thoughts = thoughts.replace(/^"|"$/g, '').trim();
  if (actions) actions = actions.replace(/^"|"$/g, '').trim();
  if (value) value = value.replace(/^"|"$/g, '').trim();
  
  return { thoughts, actions, value, raw: text };
}

// V16新增：医学报告辅助函数
function formatMetricValue(value) {
  if (value === null || value === undefined) return '-';
  if (typeof value === 'number') {
    return Number.isInteger(value) ? value : value.toFixed(1);
  }
  return value;
}

function getEfStatus(value, type) {
  if (type === 'LV') {
    if (value >= 55 && value <= 70) return 'normal';
    if (value < 40) return 'error';
    if (value < 55) return 'warning';
    return 'normal';
  } else { // RV
    if (value >= 40 && value <= 65) return 'normal';
    if (value < 30) return 'error';
    if (value < 40) return 'warning';
    return 'normal';
  }
}

function getStatusLabel(status) {
  const labels = {
    'normal': 'Normal',
    'abnormal': 'Abnormal',
    'mildly_reduced': 'Mildly Reduced',
    'severely_reduced': 'Severely Reduced',
    'elevated': 'Elevated',
    'unknown': ''
  };
  return labels[status] || '';
}

function getSectionIcon(sectionName) {
  if (sectionName.includes('Classification')) return '🏷️';
  if (sectionName.includes('LGE')) return '🔬';
  if (sectionName.includes('Left Ventricle')) return '❤️';
  if (sectionName.includes('Right Ventricle')) return '💙';
  if (sectionName.includes('Chamber') || sectionName.includes('Dimension')) return '📐';
  if (sectionName.includes('Wall Thickness') && sectionName.includes('LV')) return '🧱';
  if (sectionName.includes('Wall Thickness') && sectionName.includes('RV')) return '🧱';
  return '📊';
}

// V23: 报告 section 折叠/展开
function toggleReportSection(toggleEl) {
  const body = toggleEl.nextElementSibling;
  if (body) {
    toggleEl.classList.toggle('collapsed');
    body.classList.toggle('collapsed');
  }
}


// ========== V26: Agent VQA 优化展示 ==========

/**
 * 检测VQA回答是否为分类格式（Key: Value 或 Key: Value; Key: Value; ...）
 * 单个 key:value 也视为分类格式（如 "Tricuspid Valve: Normal"）
 */
function isCategorizedVQA(text) {
  if (!text || typeof text !== 'string') return false;
  const trimmed = text.trim();
  const parts = trimmed.split(';').map(s => s.trim()).filter(Boolean);
  if (parts.length === 0) return false;
  let kvCount = 0;
  for (const part of parts) {
    if (/^[^:]+:\s*.+/.test(part)) kvCount++;
  }
  if (parts.length === 1) {
    return kvCount === 1 && trimmed.length < 200 && !trimmed.includes('\n');
  }
  return kvCount >= 2 && kvCount / parts.length >= 0.7;
}

/**
 * 解析分类VQA文本，返回 [{key, value}] 列表
 */
function parseCategorizedVQA(text) {
  const parts = text.split(';').map(s => s.trim()).filter(Boolean);
  const items = [];
  for (const part of parts) {
    const idx = part.indexOf(':');
    if (idx > 0) {
      items.push({
        key: part.substring(0, idx).trim(),
        value: part.substring(idx + 1).trim()
      });
    }
  }
  return items;
}

/**
 * 将分类VQA项目按语义分组
 */
function groupVQAItems(items) {
  const groups = {};
  const groupDefs = [
    { pattern: /Valve/i, label: 'Valve Assessment', icon: '🫀' },
    { pattern: /Left Ventricle|Left Ventricular|^LV/i, label: 'Left Ventricle', icon: '❤️' },
    { pattern: /Right Ventricle|Right Ventricular|^RV/i, label: 'Right Ventricle', icon: '💙' },
    { pattern: /Pericardi/i, label: 'Pericardium', icon: '🛡️' },
    { pattern: /Atri/i, label: 'Atrium', icon: '💜' },
    { pattern: /Aort/i, label: 'Aorta', icon: '🔴' },
  ];

  for (const item of items) {
    let placed = false;
    for (const def of groupDefs) {
      if (def.pattern.test(item.key)) {
        if (!groups[def.label]) {
          groups[def.label] = { icon: def.icon, items: [] };
        }
        groups[def.label].items.push(item);
        placed = true;
        break;
      }
    }
    if (!placed) {
      if (!groups['Other Findings']) {
        groups['Other Findings'] = { icon: '📋', items: [] };
      }
      groups['Other Findings'].items.push(item);
    }
  }
  return groups;
}

/**
 * 判断VQA值的状态等级
 */
function getVQAValueStatus(value) {
  const v = value.toLowerCase().trim();
  const normalPatterns = /^(normal|no effusion|none|no |absent|unremarkable|preserved|stable)/i;
  const severePatterns = /(severely|severe|significantly|markedly|decreased|reduced|dilated|enlarged|thickened|thinned|impaired)/i;
  const abnormalPatterns = /(mildly|mild|slight|borderline|trace|minimal|moderate|weakened)/i;

  if (normalPatterns.test(v)) return 'normal';
  if (severePatterns.test(v)) return 'severe';
  if (abnormalPatterns.test(v)) return 'abnormal';
  return 'info';
}

/**
 * 渲染分类VQA结果（卡片网格 + 分组 + 状态摘要）
 */
function renderCategorizedVQA(items) {
  const groups = groupVQAItems(items);

  let normalCount = 0, abnormalCount = 0, severeCount = 0;
  items.forEach(it => {
    const s = getVQAValueStatus(it.value);
    if (s === 'normal') normalCount++;
    else if (s === 'abnormal') abnormalCount++;
    else if (s === 'severe') severeCount++;
  });

  let html = '<div class="vqa-categorized">';

  for (const [groupName, groupData] of Object.entries(groups)) {
    html += '<div class="vqa-group">';
    html += `<div class="vqa-group-title">${groupData.icon} ${groupName}</div>`;
    html += '<div class="vqa-items-grid">';
    for (const item of groupData.items) {
      const status = getVQAValueStatus(item.value);
      html += `
        <div class="vqa-item">
          <span class="vqa-item-name">${escapeHtml(item.key)}</span>
          <span class="vqa-item-badge status-${status}">${escapeHtml(item.value)}</span>
        </div>
      `;
    }
    html += '</div></div>';
  }

  html += `
    <div class="vqa-summary-bar">
      <span class="vqa-summary-chip"><span class="dot normal"></span> <span class="count normal">${normalCount}</span> Normal</span>
      ${abnormalCount > 0 ? `<span class="vqa-summary-chip"><span class="dot warning"></span> <span class="count warning">${abnormalCount}</span> Mildly Abnormal</span>` : ''}
      ${severeCount > 0 ? `<span class="vqa-summary-chip"><span class="dot error"></span> <span class="count error">${severeCount}</span> Abnormal</span>` : ''}
      <span class="vqa-summary-chip">${items.length} total</span>
    </div>
  `;
  html += '</div>';
  return html;
}

/**
 * 渲染纯文本VQA回答（短回答 和 长段落 使用统一容器）
 */
function renderPlainTextVQA(text) {
  const cleanText = text.trim();
  return `
    <div class="vqa-text-card">
      <div class="vqa-text-body"><p>${escapeHtml(cleanText).replace(/\n/g, '<br>')}</p></div>
    </div>
  `;
}

/**
 * 统一VQA渲染入口：自动区分分类/纯文本
 */
function renderVQAContent(text) {
  if (!text) return '';
  if (isCategorizedVQA(text)) {
    const items = parseCategorizedVQA(text);
    if (items.length >= 1) {
      return renderCategorizedVQA(items);
    }
  }
  return renderPlainTextVQA(text);
}

function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// 定期检查服务状态
setInterval(checkServerStatus, 30000);

// ========== 会话管理 ==========
async function createNewSession() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/session/create`, {
      method: 'POST'
    });
    const data = await response.json();
    currentSessionId = data.session_id;
    document.getElementById('currentSessionId').textContent = currentSessionId;
    console.log('New session created:', currentSessionId);
  } catch (error) {
    console.error('Failed to create session:', error);
    document.getElementById('currentSessionId').textContent = 'Creation failed';
  }
}

async function clearCurrentSession() {
  if (!currentSessionId) {
    alert('No active session');
    return;
  }
  
  if (!confirm('Are you sure you want to clear all cache for current session?')) {
    return;
  }
  
  try {
    await fetch(`${API_BASE_URL}/api/session/${currentSessionId}`, {
      method: 'DELETE'
    });
    // Create new session
    await createNewSession();
    // Clear chat
    clearChat();
    alert('Cache cleared, new session created');
  } catch (error) {
    console.error('Failed to clear cache:', error);
    alert('Failed to clear cache: ' + error.message);
  }
}

// ========== 图片模态框 ==========
function openImageModal(imageUrl, filename, frameIndex) {
  const modal = document.getElementById('imageModal');
  const modalImage = document.getElementById('modalImage');
  const modalInfo = document.getElementById('modalInfo');
  
  modalImage.src = imageUrl;
  modalInfo.textContent = `File: ${filename} | Frame Index: ${frameIndex}`;
  modal.classList.add('active');
  
  // 禁止背景滚动
  document.body.style.overflow = 'hidden';
}

function closeImageModal() {
  const modal = document.getElementById('imageModal');
  modal.classList.remove('active');
  document.body.style.overflow = '';
}

// ESC键关闭模态框
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    closeImageModal();
  }
});
