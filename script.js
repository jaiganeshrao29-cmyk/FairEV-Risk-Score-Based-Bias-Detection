import { GoogleGenerativeAI } from "@google/generative-ai";
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";

// --- Firebase Configuration ---
const firebaseConfig = {
  apiKey: "AIzaSyBEJ82Bro0kmzZJzX1YTyw7ZBvYuj4OuXI",
  authDomain: "ganesh-9b26a.firebaseapp.com",
  projectId: "ganesh-9b26a",
  storageBucket: "ganesh-9b26a.firebasestorage.app",
  messagingSenderId: "326813936650",
  appId: "1:326813936650:web:133bc2ffde3d754faec3f3",
  measurementId: "G-J9D1QRQ3P6"
};

// Initialize Firebase
const firebaseApp = initializeApp(firebaseConfig);
const analytics = getAnalytics(firebaseApp);

// --- State Management ---
const state = {
    applicants: [],
    weights: {
        income: 0.4,
        price: 0.3,
        location: 0.2,
        subsidy: 0.1
    },
    metrics: {
        urbanRate: 0,
        ruralRate: 0,
        biasGap: 0,
        fairnessScore: 0
    },
    charts: {
        risk: null,
        approval: null
    }
};

const API_KEY = "AIzaSyAaNDfTH738HASgYn5FPU6uJa424buJqCA";
const genAI = new GoogleGenerativeAI(API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    initApp();
});

function initApp() {
    setupNavigation();
    setupTabs();
    setupSliders();
    generateDataset();
    updateApp();
    setupAIChat();
    setupAudit();
}

// --- Navigation ---
function setupNavigation() {
    const navItems = document.querySelectorAll('.nav-links li');
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const sectionId = item.getAttribute('data-section');
            switchSection(sectionId);
            
            navItems.forEach(i => i.classList.remove('active'));
            item.classList.add('active');
        });
    });
}

function switchSection(id) {
    document.querySelectorAll('section').forEach(s => s.classList.remove('active'));
    document.getElementById(id).classList.add('active');
}

// --- Data Generation & Logic ---
function generateDataset(n = 30) {
    const applicants = [];
    const locations = ["Urban", "Rural"];
    
    for (let i = 1; i <= n; i++) {
        applicants.push({
            id: `APP${String(i).padStart(3, '0')}`,
            income: Math.floor(Math.random() * (150000 - 20000 + 1)) + 20000,
            location: locations[Math.random() > 0.6 ? 1 : 0],
            price: Math.floor(Math.random() * (80000 - 25000 + 1)) + 25000,
            subsidy: Math.random() > 0.8 ? 1 : 0
        });
    }
    state.applicants = applicants;
}

function calculateScores() {
    const { income: wInc, price: wPri, location: wLoc, subsidy: wSub } = state.weights;
    
    const incomes = state.applicants.map(a => a.income);
    const minInc = Math.min(...incomes);
    const maxInc = Math.max(...incomes);
    
    const prices = state.applicants.map(a => a.price);
    const minPri = Math.min(...prices);
    const maxPri = Math.max(...prices);
    
    state.applicants.forEach(a => {
        const normInc = (a.income - minInc) / (maxInc - minInc);
        const normPri = (a.price - minPri) / (maxPri - minPri);
        const normLoc = a.location === "Urban" ? 1 : 0;
        const normSub = a.subsidy;
        
        let score = (wInc * normInc) + (wPri * normPri) + (wLoc * normLoc) + (wSub * normSub);
        const maxPossible = wInc + wPri + wLoc + wSub;
        
        a.riskScore = maxPossible === 0 ? 0 : (score / maxPossible) * 100;
        a.decision = a.riskScore < 50 ? "Approved" : "Rejected";
    });
}

function calculateFairnessMetrics() {
    const urban = state.applicants.filter(a => a.location === "Urban");
    const rural = state.applicants.filter(a => a.location === "Rural");
    
    const uRate = urban.length > 0 ? urban.filter(a => a.decision === "Approved").length / urban.length : 0;
    const rRate = rural.length > 0 ? rural.filter(a => a.decision === "Approved").length / rural.length : 0;
    
    const gap = Math.abs(uRate - rRate);
    const score = Math.max(0, (1 - gap) * 100);
    
    state.metrics = {
        urbanRate: uRate,
        ruralRate: rRate,
        biasGap: gap,
        fairnessScore: score
    };
}

// --- UI Updates ---
function updateApp() {
    calculateScores();
    calculateFairnessMetrics();
    renderTable();
    renderMetrics();
    renderCharts();
    updateSimulationResults();
}

function renderTable() {
    const tbody = document.querySelector('#applicantsTable tbody');
    tbody.innerHTML = '';
    
    state.applicants.forEach(a => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${a.id}</td>
            <td>$${a.income.toLocaleString()}</td>
            <td>${a.location}</td>
            <td>$${a.price.toLocaleString()}</td>
            <td>${a.subsidy ? 'Yes' : 'No'}</td>
            <td>${a.riskScore.toFixed(1)}</td>
            <td class="${a.decision === 'Approved' ? 'decision-approved' : 'decision-rejected'}">${a.decision}</td>
        `;
        tbody.appendChild(row);
    });
}

function renderMetrics() {
    const { urbanRate, ruralRate, biasGap, fairnessScore } = state.metrics;
    
    document.getElementById('fairnessScoreMetric').innerText = `${fairnessScore.toFixed(1)}/100`;
    document.getElementById('biasGapMetric').innerText = biasGap.toFixed(2);
    document.getElementById('urbanRateMetric').innerText = `${(urbanRate * 100).toFixed(1)}%`;
    document.getElementById('ruralRateMetric').innerText = `${(ruralRate * 100).toFixed(1)}%`;
    
    const alertEl = document.getElementById('biasAlert');
    if (biasGap > 0.20) {
        alertEl.className = 'alert alert-error';
        alertEl.innerText = `⚠️ Bias Detected! The Bias Gap is ${biasGap.toFixed(2)} (>0.20). Mitigation required.`;
    } else {
        alertEl.className = 'alert alert-success';
        alertEl.innerText = `✅ System is Fair! The Bias Gap is ${biasGap.toFixed(2)} (≤0.20).`;
    }
}

function renderCharts() {
    const ctxRisk = document.getElementById('riskChart').getContext('2d');
    const ctxApp = document.getElementById('approvalChart').getContext('2d');
    
    // Destroy existing charts
    if (state.charts.risk) state.charts.risk.destroy();
    if (state.charts.approval) state.charts.approval.destroy();
    
    // Risk Histogram (Approx)
    const bins = Array(10).fill(0);
    state.applicants.forEach(a => {
        const binIndex = Math.min(Math.floor(a.riskScore / 10), 9);
        bins[binIndex]++;
    });
    
    state.charts.risk = new Chart(ctxRisk, {
        type: 'bar',
        data: {
            labels: ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100'],
            datasets: [{
                label: 'Applicant Count',
                data: bins,
                backgroundColor: '#3b82f6',
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.1)' } },
                x: { grid: { display: false } }
            },
            plugins: { legend: { display: false } }
        }
    });
    
    // Approval Bar Chart
    state.charts.approval = new Chart(ctxApp, {
        type: 'bar',
        data: {
            labels: ['Urban', 'Rural'],
            datasets: [{
                data: [state.metrics.urbanRate * 100, state.metrics.ruralRate * 100],
                backgroundColor: ['#8b5cf6', '#10b981'],
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { beginAtZero: true, max: 100, grid: { color: 'rgba(255,255,255,0.1)' } },
                x: { grid: { display: false } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

// --- Tabs & UI ---
function setupTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.getAttribute('data-tab');
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(target).classList.add('active');
            
            if (target === 'scoring-logic') renderWeights();
        });
    });
}

function renderWeights() {
    const grid = document.getElementById('currentWeights');
    grid.innerHTML = '';
    const labels = { income: 'Income', price: 'Vehicle Price', location: 'Location', subsidy: 'Prev Subsidy' };
    
    Object.entries(state.weights).forEach(([key, val]) => {
        const card = document.createElement('div');
        card.className = 'metric-card';
        card.innerHTML = `<label>${labels[key]} Weight</label><span>${val.toFixed(2)}</span>`;
        grid.appendChild(card);
    });
}

// --- Simulation ---
function setupSliders() {
    const sliders = ['w_income', 'w_price', 'w_location', 'w_subsidy'];
    sliders.forEach(id => {
        const el = document.getElementById(id);
        el.addEventListener('input', () => {
            const key = id.split('_')[1];
            state.weights[key] = parseFloat(el.value);
            document.getElementById(`val_${key}`).innerText = el.value;
            updateSimulationResults();
        });
    });
    
    document.getElementById('resetWeightsBtn').addEventListener('click', () => {
        state.weights = { income: 0.4, price: 0.3, location: 0.2, subsidy: 0.1 };
        sliders.forEach(id => {
            const key = id.split('_')[1];
            document.getElementById(id).value = state.weights[key];
            document.getElementById(`val_${key}`).innerText = state.weights[key];
        });
        updateApp();
    });
}

function updateSimulationResults() {
    // We calculate a temporary copy for the "Simulation" values if we haven't clicked "Deploy"
    // In this static version, I'll make the simulation real-time globally.
    updateApp(); // Simplified for the static prototype
    
    document.getElementById('simFairness').innerText = `${state.metrics.fairnessScore.toFixed(1)}/100`;
    document.getElementById('simBiasGap').innerText = state.metrics.biasGap.toFixed(2);
}

// --- AI Features ---
async function setupAIChat() {
    const btn = document.getElementById('sendChatBtn');
    const input = document.getElementById('chatInput');
    const history = document.getElementById('chatHistory');
    
    const appendMessage = (role, text) => {
        const bubble = document.createElement('div');
        bubble.className = `chat-bubble ${role}-bubble`;
        bubble.innerText = text;
        history.appendChild(bubble);
        history.scrollTop = history.scrollHeight;
    };
    
    appendMessage('ai', "Hello! I am the FairEV Assistant. Ask me anything about AI bias or fairness metrics.");

    btn.addEventListener('click', async () => {
        const text = input.value.trim();
        if (!text) return;
        
        appendMessage('user', text);
        input.value = '';
        
        try {
            const result = await model.generateContent(text);
            const response = await result.response;
            appendMessage('ai', response.text());
        } catch (err) {
            appendMessage('ai', "Error: " + err.message);
        }
    });
}

async function setupAudit() {
    const btn = document.getElementById('generateAuditBtn');
    const output = document.getElementById('auditReportOutput');
    
    btn.addEventListener('click', async () => {
        output.innerText = "Analyzing metrics and generating report...";
        const { urbanRate, ruralRate, biasGap, fairnessScore } = state.metrics;
        
        const prompt = `
            You are an expert AI Auditor assessing an EV subsidy allocation system.
            Current Metrics:
            - Urban Approval Rate: ${(urbanRate*100).toFixed(1)}%
            - Rural Approval Rate: ${(ruralRate*100).toFixed(1)}%
            - Bias Gap: ${biasGap.toFixed(2)} (Acceptable Threshold: <0.20)
            - Overall Fairness Score: ${fairnessScore.toFixed(1)}/100
            
            Please provide a professional, structured report covering:
            1. **Executive Summary**: Is the system currently fair?
            2. **Root Cause Analysis**: Why the location feature might be skewing results.
            3. **Actionable Recommendations**: 3 concise steps to mitigate this bias.
            Use clear headings and bullet points.
        `;
        
        try {
            const result = await model.generateContent(prompt);
            const response = await result.response;
            output.innerHTML = formatMarkdown(response.text());
        } catch (err) {
            output.innerText = "Error: " + err.message;
        }
    });
}

// Simple markdown formatter helper
function formatMarkdown(text) {
    return text
        .replace(/### (.*)/g, '<h3>$1</h3>')
        .replace(/\*\* (.*)\*\*/g, '<b>$1</b>')
        .replace(/\* (.*)/g, '<li>$1</li>')
        .replace(/\n/g, '<br>');
}
