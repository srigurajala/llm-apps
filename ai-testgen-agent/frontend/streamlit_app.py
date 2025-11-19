"""
AI Test Case Generator - Streamlit Frontend
============================================
Professional UI for agentic test generation
"""

import streamlit as st
import requests
import json
from datetime import datetime
import time
import os

# Page configuration
st.set_page_config(
    page_title="AI Test Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #8b5cf6;
        --secondary-color: #ec4899;
        --success-color: #10b981;
        --error-color: #ef4444;
        --warning-color: #f59e0b;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    /* Card styling */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #8b5cf6;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #8b5cf6;
        margin: 0;
    }
    
    .metric-label {
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }
    
    /* Code block styling */
    .code-container {
        background: #1e293b;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .badge-success {
        background: #d1fae5;
        color: #065f46;
    }
    
    .badge-error {
        background: #fee2e2;
        color: #991b1b;
    }
    
    .badge-info {
        background: #dbeafe;
        color: #1e40af;
    }
    
    .badge-warning {
        background: #fef3c7;
        color: #92400e;
    }
    
    /* Log entry styling */
    .log-entry {
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        border-left: 3px solid;
    }
    
    .log-info {
        background: #eff6ff;
        border-color: #3b82f6;
        color: #1e40af;
    }
    
    .log-success {
        background: #f0fdf4;
        border-color: #10b981;
        color: #065f46;
    }
    
    .log-error {
        background: #fef2f2;
        border-color: #ef4444;
        color: #991b1b;
    }
    
    .log-system {
        background: #faf5ff;
        border-color: #8b5cf6;
        color: #6b21a8;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Remove default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Test result styling */
    .test-passed {
        background: #d1fae5;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #10b981;
        margin: 0.5rem 0;
    }
    
    .test-failed {
        background: #fee2e2;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ef4444;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generated_code' not in st.session_state:
    st.session_state.generated_code = ""
if 'test_results' not in st.session_state:
    st.session_state.test_results = None
if 'agent_logs' not in st.session_state:
    st.session_state.agent_logs = []
if 'history' not in st.session_state:
    st.session_state.history = []
if 'backend_url' not in st.session_state:
    st.session_state.backend_url = "http://localhost:8000"

# Helper functions
def add_log(message, log_type="info"):
    """Add a log entry with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.agent_logs.append({
        "timestamp": timestamp,
        "message": message,
        "type": log_type
    })

def check_backend_health():
    """Check if backend is available"""
    try:
        response = requests.get(f"{st.session_state.backend_url}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_system_info():
    """Get system information from backend"""
    try:
        response = requests.get(f"{st.session_state.backend_url}/system-info", timeout=2)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

# Main header
st.markdown("""
<div class="main-header">
    <h1>🤖 Agentic AI Test Generator</h1>
    <p>Autonomous test case generation powered by LangChain & Open Source LLMs</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Backend URL
    backend_url = st.text_input(
        "Backend URL",
        value=st.session_state.backend_url,
        help="URL of the FastAPI backend"
    )
    st.session_state.backend_url = backend_url
    
    # Check backend status
    backend_status = check_backend_health()
    if backend_status:
        st.success("✅ Backend Connected")
        
        # Get system info
        system_info = get_system_info()
        if system_info:
            st.markdown("---")
            st.markdown("### 🔧 System Info")
            st.info(f"**Provider:** {system_info.get('llm_provider', 'Unknown')}")
            st.info(f"**Model:** {system_info.get('llm_model', 'Unknown')}")
            st.info(f"**Status:** {system_info.get('status', 'Unknown')}")
    else:
        st.error("❌ Backend Offline")
        st.warning("Please start the FastAPI backend")
    
    st.markdown("---")
    
    # Framework selection
    framework = st.selectbox(
        "Testing Framework",
        ["jest", "mocha", "jasmine", "vitest"],
        help="Select your preferred testing framework"
    )
    
    # Advanced options
    st.markdown("### 🎯 Options")
    include_edge_cases = st.checkbox("Include Edge Cases", value=True)
    cicd_enabled = st.checkbox("Enable CI/CD Integration", value=False)
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("### 🚀 Quick Actions")
    
    if st.button("🔄 Clear History"):
        st.session_state.history = []
        st.session_state.agent_logs = []
        st.rerun()
    
    if st.button("📋 Clear Logs"):
        st.session_state.agent_logs = []
        st.rerun()
    
    st.markdown("---")
    
    # Help section
    with st.expander("ℹ️ Help & Examples"):
        st.markdown("""
        **Example prompts:**
        - Login page should validate email format
        - Shopping cart should calculate total with tax
        - API should handle rate limiting
        - Form should validate required fields
        
        **Supported frameworks:**
        - Jest (React, Node.js)
        - Mocha (Node.js)
        - Jasmine (Angular)
        - Vitest (Vite projects)
        """)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📝 Feature Description")
    
    # Feature input
    description = st.text_area(
        "Describe the feature you want to test",
        placeholder="Example: The login page should reject wrong passwords and display an error message.",
        height=150,
        help="Describe your feature in natural language"
    )
    
    # Generate button
    if st.button("🎯 Generate Tests with AI Agent", type="primary", disabled=not backend_status):
        if description.strip():
            st.session_state.agent_logs = []
            st.session_state.generated_code = ""
            st.session_state.test_results = None
            
            add_log("🚀 Starting AI Agent...", "system")
            add_log("📝 Analyzing feature description...", "info")
            
            with st.spinner("AI Agent is working..."):
                try:
                    response = requests.post(
                        f"{st.session_state.backend_url}/generate",
                        json={
                            "description": description,
                            "framework": framework,
                            "language": "javascript",
                            "include_edge_cases": include_edge_cases,
                            "cicd_enabled": cicd_enabled
                        },
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.generated_code = data['code']
                        
                        add_log("✅ Code analysis complete", "success")
                        add_log("🔧 Test cases generated successfully", "success")
                        add_log(f"📊 Generated {data['metadata']['test_count']} test cases", "info")
                        
                        # Add to history
                        st.session_state.history.insert(0, {
                            "description": description,
                            "framework": framework,
                            "code": data['code'],
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "metadata": data['metadata']
                        })
                        
                        st.success("✨ Test generation complete!")
                        st.rerun()
                    else:
                        add_log(f"❌ Error: {response.text}", "error")
                        st.error(f"Generation failed: {response.text}")
                        
                except Exception as e:
                    add_log(f"❌ Error: {str(e)}", "error")
                    st.error(f"Connection error: {str(e)}")
        else:
            st.warning("⚠️ Please enter a feature description")
    
    # Agent activity log
    st.markdown("---")
    st.markdown("### 📊 Agent Activity")
    
    log_container = st.container()
    with log_container:
        if st.session_state.agent_logs:
            for log in st.session_state.agent_logs[-10:]:  # Show last 10 logs
                log_class = f"log-{log['type']}"
                icon = {
                    "info": "ℹ️",
                    "success": "✅",
                    "error": "❌",
                    "system": "🤖"
                }.get(log['type'], "•")
                
                st.markdown(f"""
                <div class="log-entry {log_class}">
                    {icon} <strong>[{log['timestamp']}]</strong> {log['message']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Waiting for agent activity...")
    
    # History section
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Recent Generations")
        
        for idx, entry in enumerate(st.session_state.history[:5]):
            with st.expander(f"{entry['framework']} - {entry['description'][:50]}..."):
                st.caption(f"🕐 {entry['timestamp']}")
                st.caption(f"📊 {entry['metadata']['test_count']} tests")
                if st.button(f"Load this code", key=f"load_{idx}"):
                    st.session_state.generated_code = entry['code']
                    st.rerun()

with col2:
    st.markdown("### 💻 Generated Test Code")
    
    if st.session_state.generated_code:
        # Display code with syntax highlighting
        st.code(st.session_state.generated_code, language="javascript", line_numbers=True)
        
        # Action buttons
        button_col1, button_col2, button_col3 = st.columns(3)
        
        with button_col1:
            if st.button("▶️ Run Tests"):
                add_log("🧪 Executing tests...", "info")
                
                with st.spinner("Running tests..."):
                    try:
                        response = requests.post(
                            f"{st.session_state.backend_url}/execute",
                            json={
                                "code": st.session_state.generated_code,
                                "framework": framework
                            },
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            st.session_state.test_results = response.json()
                            add_log(f"✅ Tests complete: {st.session_state.test_results['passed']}/{st.session_state.test_results['total']} passed", "success")
                            st.rerun()
                        else:
                            add_log("❌ Test execution failed", "error")
                            st.error("Test execution failed")
                    except Exception as e:
                        add_log(f"❌ Error: {str(e)}", "error")
                        st.error(f"Error: {str(e)}")
        
        with button_col2:
            # Download button
            st.download_button(
                label="📥 Download",
                data=st.session_state.generated_code,
                file_name=f"test.{framework}.js",
                mime="text/javascript"
            )
        
        with button_col3:
            if cicd_enabled and st.button("🔄 Get CI/CD"):
                add_log("🔄 Generating CI/CD configuration...", "info")
                
                with st.spinner("Generating CI/CD config..."):
                    try:
                        response = requests.post(
                            f"{st.session_state.backend_url}/cicd-config",
                            params={
                                "framework": framework,
                                "code": st.session_state.generated_code
                            },
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            add_log("✅ CI/CD config generated", "success")
                            
                            st.download_button(
                                label="📥 Download CI/CD Config",
                                data=data['config'],
                                file_name=".github/workflows/test.yml",
                                mime="text/yaml"
                            )
                        else:
                            st.error("CI/CD generation failed")
                    except Exception as e:
                        add_log(f"❌ Error: {str(e)}", "error")
                        st.error(f"Error: {str(e)}")
        
        # Test results section
        if st.session_state.test_results:
            st.markdown("---")
            st.markdown("### 📊 Test Results")
            
            results = st.session_state.test_results
            
            # Metrics
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{results['total']}</div>
                    <div class="metric-label">Total Tests</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col2:
                st.markdown(f"""
                <div class="metric-card" style="border-color: #10b981;">
                    <div class="metric-value" style="color: #10b981;">{results['passed']}</div>
                    <div class="metric-label">Passed</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col3:
                st.markdown(f"""
                <div class="metric-card" style="border-color: #ef4444;">
                    <div class="metric-value" style="color: #ef4444;">{results['failed']}</div>
                    <div class="metric-label">Failed</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="font-size: 1.5rem;">{results['duration']}</div>
                    <div class="metric-label">Duration</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Individual test results
            st.markdown("#### Test Cases")
            for test in results['tests']:
                if test['status'] == 'passed':
                    st.markdown(f"""
                    <div class="test-passed">
                        ✅ <strong>{test['name']}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="test-failed">
                        ❌ <strong>{test['name']}</strong>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("👈 Enter a feature description and click 'Generate Tests' to see the code here")
        
        # Show example
        with st.expander("💡 See Example Output"):
            st.code("""describe('login_validation', () => {
  it('should accept valid credentials', () => {
    const result = login('user@example.com', 'password123');
    expect(result.success).toBe(true);
  });

  it('should reject invalid email format', () => {
    expect(() => login('invalid-email', 'password123'))
      .toThrow('Invalid email format');
  });

  it('should reject empty password', () => {
    expect(() => login('user@example.com', ''))
      .toThrow('Password required');
  });
});""", language="javascript")

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**🤖 Powered by:**")
    st.markdown("LangChain • Open Source LLMs")

with footer_col2:
    st.markdown("**📚 Documentation:**")
    st.markdown("[GitHub](https://github.com) • [Docs](https://docs.langchain.com)")

with footer_col3:
    st.markdown("**💡 Tips:**")
    st.markdown("Use clear descriptions • Include requirements")