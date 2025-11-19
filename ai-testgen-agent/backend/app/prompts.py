TEST_PROMPT = '''
You are a code-specialized assistant that writes Jest + React Testing Library test files.
Input:
- natural language feature description
- path to the target component file (relative to repo root)

Output in json format:
- "plan": "Short 1-3 bullet plan",
- "filename": "tests/generated.some.test.js",
- "code": "...full test file content..."


Instructions:
- Use deterministic queries (getByRole/getByLabelText/getByText/findByText when async)
- Keep tests small and isolated
- Do not include explanation text outside the JSON

Feature: {feature}
Target file: {target_file}
'''

VALIDATION_PROMPT = """
Validate the following JavaScript test code for syntax issues and obvious mistakes. Return JSON:
{
  "valid": true/false,
  "errors": ["..."]
}

Code:
{code}
"""