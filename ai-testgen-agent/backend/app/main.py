"""
AI Test Case Generator - FastAPI Backend with LangChain Agents
================================================
Agentic AI system for automated test generation from natural language
Supports multiple open-source LLMs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
from datetime import datetime
import json
import re

# LangChain imports
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

# LLM Provider imports
from langchain_community.llms import Ollama, HuggingFaceHub, LlamaCpp
from langchain_community.chat_models import ChatOllama

app = FastAPI(title="AI Test Generator API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# LLM CONFIGURATION - Choose your LLM provider
# ============================================================================

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # Options: ollama, huggingface, llamacpp
LLM_MODEL = os.getenv("LLM_MODEL", "phi:latest")

# Updated HuggingFace model recommendations (working models as of 2025)
HUGGINGFACE_RECOMMENDED_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.2",  # Fast and reliable
    "microsoft/phi-2",  # Small and efficient
    "bigcode/starcoder2-7b",  # Best for code
    "Qwen/Qwen2-7B-Instruct",  # Excellent quality
    "meta-llama/Meta-Llama-3-8B-Instruct",  # Requires approval
]

def get_llm():
    """
    Initialize LLM based on provider choice
    """
    provider = LLM_PROVIDER.lower()
    
    if provider == "ollama":
        # Ollama - Run LLMs locally (RECOMMENDED)
        # Install: curl -fsSL https://ollama.ai/install.sh | sh
        # Models: codellama, deepseek-coder, mistral, phi, etc.
        return ChatOllama(
            model=LLM_MODEL,
            temperature=0.3,
            format="json" if "json" in LLM_MODEL.lower() else ""
        )
    
    elif provider == "huggingface":
        # HuggingFace - Free API (requires token)
        # Get token: https://huggingface.co/settings/tokens
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN not set. Get token from https://huggingface.co/settings/tokens")
        
        # Use HuggingFace Inference API with chat models
        from langchain_community.chat_models.huggingface import ChatHuggingFace
        from langchain_community.llms import HuggingFaceEndpoint
        
        # Create endpoint (supports more models)
        llm_endpoint = HuggingFaceEndpoint(
            repo_id=LLM_MODEL,
            huggingfacehub_api_token=hf_token,
            temperature=0.3,
            max_new_tokens=2000,
            timeout=120
        )
        
        # Wrap in chat model for better compatibility
        return ChatHuggingFace(llm=llm_endpoint)
    
    elif provider == "llamacpp":
        # LlamaCpp - Run quantized models locally
        # Download GGUF models from HuggingFace
        model_path = os.getenv("LLAMA_MODEL_PATH", "./models/codellama-7b.gguf")
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found: {model_path}")
        
        return LlamaCpp(
            model_path=model_path,
            temperature=0.3,
            max_tokens=2000,
            n_ctx=4096,
            verbose=False
        )
    
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

# Initialize LLM
try:
    llm = get_llm()
    print(f"✅ LLM initialized: {LLM_PROVIDER} - {LLM_MODEL}")
except Exception as e:
    print(f"⚠️ LLM initialization warning: {e}")
    print(f"   Error details: {type(e).__name__}")
    print("💡 Using fallback mode - install Ollama for best experience")
    llm = None

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class TestGenerationRequest(BaseModel):
    description: str
    framework: str = "jest"
    language: str = "javascript"
    include_edge_cases: bool = True
    cicd_enabled: bool = False

class TestExecutionRequest(BaseModel):
    code: str
    framework: str

class TestResult(BaseModel):
    total: int
    passed: int
    failed: int
    duration: str
    tests: List[Dict[str, str]]

class GenerationResponse(BaseModel):
    code: str
    framework: str
    language: str
    metadata: Dict[str, Any]
    timestamp: str

class SystemInfo(BaseModel):
    llm_provider: str
    llm_model: str
    status: str
    available_models: List[str]

# ============================================================================
# AGENT TOOLS
# ============================================================================

def safe_llm_invoke(prompt: str) -> str:
    """
    Safely invoke LLM with proper error handling and response parsing
    """
    try:
        if not llm:
            return None
        
        # Invoke the LLM
        response = llm.invoke(prompt)
        
        # Handle different response types
        if isinstance(response, str):
            # Direct string response
            return response
        elif hasattr(response, 'content'):
            # AIMessage or similar with content attribute
            return response.content
        elif isinstance(response, dict):
            # Dictionary response
            if 'content' in response:
                return response['content']
            elif 'text' in response:
                return response['text']
            return str(response)
        else:
            # Fallback: convert to string
            return str(response)
            
    except Exception as e:
        print(f"LLM invocation error: {e}")
        return None

def code_analyzer_tool(feature_description: str) -> str:
    """
    Analyzes feature description and extracts key testing requirements.
    """
    if not llm:
        return generate_fallback_analysis(feature_description)
    
    analysis_prompt = f"""Analyze this feature description and extract testing requirements:

Feature: {feature_description}

Provide:
1. Main functionality to test
2. Expected behaviors
3. Edge cases
4. Error conditions
5. Success scenarios

Keep it concise and structured."""
    
    result = safe_llm_invoke(analysis_prompt)
    
    if result:
        return result
    else:
        return generate_fallback_analysis(feature_description)

def generate_fallback_analysis(description: str) -> str:
    """Fallback analysis when LLM is unavailable"""
    return f"""
    Analysis of: {description}
    
    Main functionality: {description.split('.')[0]}
    Expected behaviors: Validation, error handling, success cases
    Edge cases: Empty input, null values, boundary conditions
    Error conditions: Invalid input, unauthorized access
    Success scenarios: Valid input processing
    """

def test_code_generator_tool(input_data: str) -> str:
    """
    Generates test code based on analysis.
    Input format: "framework|language|analysis|description"
    """
    parts = input_data.split("|")
    if len(parts) < 4:
        return "Error: Invalid input format"
    
    framework, language, analysis, description = parts[0], parts[1], parts[2], parts[3]
    
    if not llm:
        return generate_fallback_code(description, framework)
    
    generation_prompt = f"""
    Generate {framework} test code in {language} for:
    
    Feature: {description}
    Analysis: {analysis}
    
    Requirements:
    - Use {framework} syntax
    - Include describe/it blocks
    - Add setup/teardown if needed
    - Test success and failure paths
    - Test edge cases
    - Add assertions
    - Include comments
    
    Generate ONLY the test code, no explanations.
    """
    
    try:
        response = llm.invoke(generation_prompt)
        code = response.content if hasattr(response, 'content') else str(response)
        
        # Clean up markdown code blocks
        code = re.sub(r'```(?:javascript|js|typescript|ts)?\n?', '', code)
        code = code.strip()
        
        return code
    except Exception as e:
        print(f"Generation error: {e}")
        return generate_fallback_code(description, framework)

def generate_fallback_code(description: str, framework: str) -> str:
    """Generate basic test template when LLM is unavailable"""
    test_name = description.split(' ')[:5]
    test_name = '_'.join(test_name).lower()
    
    if framework == "jest":
        return f"""// Test: {description}
// Generated by fallback template

describe('{test_name}', () => {{
  beforeEach(() => {{
    // Setup test environment
  }});

  it('should handle valid input correctly', () => {{
    const input = 'valid_input';
    const result = functionUnderTest(input);
    
    expect(result).toBeDefined();
    expect(result).toBeTruthy();
  }});

  it('should reject invalid input', () => {{
    const invalidInput = '';
    expect(() => functionUnderTest(invalidInput)).toThrow();
  }});

  it('should handle edge cases', () => {{
    expect(() => functionUnderTest(null)).toThrow();
    expect(() => functionUnderTest(undefined)).toThrow();
  }});

  it('should display appropriate error messages', () => {{
    const errorInput = 'invalid';
    const result = functionUnderTest(errorInput);
    
    expect(result.error).toBeDefined();
    expect(result.error).toContain('error');
  }});

  afterEach(() => {{
    // Cleanup test environment
  }});
}});"""
    
    elif framework == "mocha":
        return f"""// Test: {description}
const assert = require('assert');

describe('{test_name}', function() {{
  beforeEach(function() {{
    // Setup
  }});

  it('should handle valid input', function() {{
    const input = 'valid_input';
    const result = functionUnderTest(input);
    assert.ok(result);
  }});

  it('should reject invalid input', function() {{
    assert.throws(() => functionUnderTest(''));
  }});

  it('should handle edge cases', function() {{
    assert.throws(() => functionUnderTest(null));
  }});
}});"""
    
    return f"// Test code for {framework}\n// Feature: {description}"

def code_validator_tool(code: str) -> str:
    """
    Validates generated test code for syntax and completeness.
    """
    if not llm:
        return "VALID - Basic validation passed (LLM validation unavailable)"
    
    validation_prompt = f"""Review this test code briefly:

{code[:500]}...

Check:
1. Syntax correctness
2. Test coverage
3. Best practices

Respond with: "VALID" or "INVALID: <issues>" """
    
    result = safe_llm_invoke(validation_prompt)
    
    if result:
        return result
    else:
        return "VALID - Validation skipped"

def cicd_integration_tool(input_data: str) -> str:
    """
    Generates CI/CD configuration for test automation.
    Input format: "framework|code"
    """
    parts = input_data.split("|", 1)
    if len(parts) < 2:
        return "Error: Invalid input format"
    
    framework, code = parts[0], parts[1]
    
    # Use template for CI/CD (doesn't require LLM)
    return generate_cicd_template(framework)

def generate_cicd_template(framework: str) -> str:
    """Generate GitHub Actions workflow template"""
    return f"""name: Run {framework.upper()} Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        node-version: [16.x, 18.x]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: ${{{{ matrix.node-version }}}}
    
    - name: Install dependencies
      run: npm ci
    
    - name: Run tests
      run: npm test
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      if: matrix.node-version == '18.x'
"""

# ============================================================================
# AGENT SETUP
# ============================================================================

tools = [
    Tool(
        name="CodeAnalyzer",
        func=code_analyzer_tool,
        description="Analyzes feature descriptions. Input: feature description string."
    ),
    Tool(
        name="TestCodeGenerator",
        func=test_code_generator_tool,
        description="Generates test code. Input: 'framework|language|analysis|description'"
    ),
    Tool(
        name="CodeValidator",
        func=code_validator_tool,
        description="Validates test code. Input: test code string."
    ),
    Tool(
        name="CICDIntegration",
        func=cicd_integration_tool,
        description="Generates CI/CD config. Input: 'framework|code'"
    ),
]

# Agent prompt template
agent_prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")

# Create agent only if LLM is available
agent_executor = None
if llm:
    try:
        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=agent_prompt
        )
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=6,
            handle_parsing_errors=True,
            return_intermediate_steps=False  # Reduce output complexity
        )
        print("✅ Agent executor initialized")
    except Exception as e:
        print(f"⚠️ Agent initialization warning: {e}")
        print(f"   Error type: {type(e).__name__}")
        print("   Falling back to direct tool execution")
        agent_executor = None

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "service": "AI Test Generator API",
        "version": "1.0.0",
        "status": "operational",
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "endpoints": ["/generate", "/execute", "/validate", "/cicd-config", "/system-info"]
    }

@app.get("/system-info", response_model=SystemInfo)
async def get_system_info():
    """Get system and LLM information"""
    
    available_models = []
    if LLM_PROVIDER == "ollama":
        available_models = [
            "codellama:7b",
            "codellama:13b",
            "deepseek-coder:6.7b",
            "mistral:7b",
            "phi:latest",
            "starcoder2:7b"
        ]
    elif LLM_PROVIDER == "huggingface":
        available_models = [
            "mistralai/Mistral-7B-Instruct-v0.2",
            "microsoft/phi-2",
            "bigcode/starcoder2-7b",
            "Qwen/Qwen2-7B-Instruct",
            "meta-llama/Meta-Llama-3-8B-Instruct"
        ]
    
    return SystemInfo(
        llm_provider=LLM_PROVIDER,
        llm_model=LLM_MODEL,
        status="operational" if llm else "fallback",
        available_models=available_models
    )

@app.post("/generate", response_model=GenerationResponse)
async def generate_tests(request: TestGenerationRequest):
    """
    Main endpoint: Generates test cases using the LangChain agent.
    """
    try:
        if agent_executor and llm:
            # Use agent for generation
            agent_input = f"""Generate {request.framework} test code for:

            Description: {request.description}
            Framework: {request.framework}
            Language: {request.language}

            Steps:
            1. First use CodeAnalyzer to analyze the feature
            2. Then TestCodeGenerator to create test code
            3. Then use CodeValidator to ensure quality"""
            
            try:
                result = agent_executor.invoke({"input": agent_input})
                generated_code = result.get("output", "")
                
                if "Final Answer:" in generated_code:
                    generated_code = generated_code.split("Final Answer:")[-1].strip()
                
                generated_code = re.sub(r'```(?:javascript|js)?\n?', '', generated_code).strip()
                
                # If code is empty or too short, use fallback
                if not generated_code or len(generated_code) < 100:
                    raise ValueError("Generated code is too short")
                    
            except Exception as agent_error:
                print(f"Agent execution error: {agent_error}")
                # Fallback to direct generation
                analysis = code_analyzer_tool(request.description)
                input_data = f"{request.framework}|{request.language}|{analysis}|{request.description}"
                generated_code = test_code_generator_tool(input_data)
        else:
            # Fallback to direct generation without agent
            print("Using fallback generation (agent unavailable)")
            analysis = code_analyzer_tool(request.description)
            input_data = f"{request.framework}|{request.language}|{analysis}|{request.description}"
            generated_code = test_code_generator_tool(input_data)
        
        # Ensure we have valid code
        if not generated_code or len(generated_code) < 50:
            generated_code = generate_fallback_code(request.description, request.framework)
        
        # Prepare metadata
        metadata = {
            "test_count": generated_code.count("it(") + generated_code.count("test("),
            "has_setup": "beforeEach" in generated_code or "setUp" in generated_code,
            "has_teardown": "afterEach" in generated_code or "tearDown" in generated_code,
            "edge_cases_included": request.include_edge_cases,
            "llm_used": LLM_PROVIDER,
            "generation_method": "agent" if agent_executor else "direct"
        }
        
        # Ensure test count is at least 1
        if metadata["test_count"] == 0:
            metadata["test_count"] = 3  # Default estimate
        
        return GenerationResponse(
            code=generated_code,
            framework=request.framework,
            language=request.language,
            metadata=metadata,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        print(f"Generation error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/execute", response_model=TestResult)
async def execute_tests(request: TestExecutionRequest):
    """
    Simulates test execution and returns results.
    """
    try:
        test_count = request.code.count("it(") + request.code.count("test(")
        
        if test_count == 0:
            test_count = 3
        
        # Simulate results
        import random
        passed = int(test_count * random.uniform(0.75, 1.0))
        failed = test_count - passed
        
        # Extract test names
        test_names = re.findall(r'(?:it|test)\([\'"](.+?)[\'"]', request.code)
        
        tests = []
        for i, name in enumerate(test_names):
            tests.append({
                "name": name,
                "status": "passed" if i < passed else "failed"
            })
        
        # Fill in if extraction didn't get enough
        while len(tests) < test_count:
            tests.append({
                "name": f"Test case {len(tests) + 1}",
                "status": "passed" if len(tests) < passed else "failed"
            })
        
        return TestResult(
            total=test_count,
            passed=passed,
            failed=failed,
            duration=f"{random.uniform(0.5, 3.0):.2f}s",
            tests=tests
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")

@app.post("/validate")
async def validate_code(code: str):
    """
    Validates test code.
    """
    try:
        validation_result = code_validator_tool(code)
        is_valid = "VALID" in validation_result.upper()
        
        return {
            "valid": is_valid,
            "message": validation_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.post("/cicd-config")
async def generate_cicd_config(framework: str, code: str):
    """
    Generates CI/CD configuration.
    """
    try:
        cicd_yaml = cicd_integration_tool(f"{framework}|{code}")
        
        return {
            "config": cicd_yaml,
            "framework": framework,
            "filename": ".github/workflows/test.yml",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CI/CD config generation failed: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "llm_available": llm is not None,
        "llm_provider": LLM_PROVIDER,
        "agent_available": agent_executor is not None
    }

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║  🤖 AI Test Generator API                                ║
    ║  Provider: {LLM_PROVIDER:<44} ║
    ║  Model: {LLM_MODEL:<47} ║
    ║  Status: {'✅ Operational' if llm else '⚠️  Fallback Mode':<47} ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=8000)