# backend/app/agent.py
import os
import re
import json
import tempfile
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from git import Repo

load_dotenv()

class TestGeneratorAgent:
    """
    Agent that calls CodeLlama via huggingface_hub.InferenceClient to produce
    test files from a natural-language feature description.
    """

    def __init__(self, repo_path: str, model: str = "mistralai/Mistral-Nemo-Instruct-2407"):
        self.repo_path = Path(repo_path)
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in environment")

        # Create a single InferenceClient instance (uses HF token)
        # The client will route to the HF Inference API or your configured provider
        self.client = InferenceClient(token=hf_token)
        self.model = model

        # Minimal generation parameters — tune if you have a different model
        self.generation_kwargs = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_k": 50,
            # add other TGI parameters if necessary
        }

    def _call_model(self, feature: str, target_file: str):
        """
        Call the HF Inference API text-generation endpoint for the configured model.
        We build an instruction-style prompt that requests JSON output with keys:
        plan, filename, code
        """
        prompt = (
            "You are an expert code assistant. Produce a JSON object ONLY with keys:\n"
            '"plan" (short bullet list), "filename" (relative path), and "code" (string with full file contents).\n\n'
            "Constraints: The code must be a valid Jest + React Testing Library test file. "
            "Do NOT include any explanation or markdown — return a single JSON object.\n\n"
            f"Feature: {feature}\n"
            f"Target file: {target_file}\n"
            "\nReturn the JSON now.\n"
        )

        # Use the InferenceClient text_generation helper if available. Different versions expose
        # slightly different helpers; we attempt common ones with fallbacks.
        try:
            # preferred: high-level text_generation method
            print(prompt)
            generated_text = self.client.text_generation(prompt="The quick brown fox jumps over the")
            print(generated_text)
            resp = self.client.text_generation(prompt, model=self.model, max_new_tokens=50)
            print(resp)
            # The HF inference response shape can be a list/dict — find generated text
            if isinstance(resp, dict) and "generated_text" in resp:
                text = resp["generated_text"]
            elif isinstance(resp, list):
                # often a list of dicts with 'generated_text'
                text = "".join([item.get("generated_text", "") for item in resp])
            else:
                # convert to string
                text = str(resp)
        except AttributeError:
            # fallback to generic `client.generate` / `client.inference` method names
            try:
                resp = self.client.generate(model=self.model, inputs=prompt, parameters=self.generation_kwargs)
                # similar parsing
                if isinstance(resp, dict) and "generated_text" in resp:
                    text = resp["generated_text"]
                elif isinstance(resp, list):
                    text = "".join([item.get("generated_text", "") for item in resp])
                else:
                    text = str(resp)
            except Exception as e:
                raise RuntimeError(f"Model call failed: {e}")

        return text

    def _extract_json(self, text: str):
        """
        Try to extract a JSON object from the model text output.
        Attempts a few cleaning heuristics to handle minor quoting issues.
        """
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        body = m.group(0)
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            # quick heuristic fixes: single->double quotes, remove trailing commas
            cleaned = body.replace("'", '"')
            cleaned = re.sub(r",\\s*}", "}", cleaned)
            cleaned = re.sub(r",\\s*\\]", "]", cleaned)
            try:
                return json.loads(cleaned)
            except Exception:
                return None

    def _write_test_file(self, filename: str, code: str):
        dest = self.repo_path / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(code, encoding="utf-8")
        return dest

    def _run_tests(self):
        """
        Run `npm test` inside the repo. For production, run inside a container.
        Capture stdout/stderr and return structured output.
        """
        try:
            proc = subprocess.run(
                ["npm", "test", "--", "--json", "--outputFile=jest-result.json"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=120,
            )
            result_file = self.repo_path / "jest-result.json"
            res = {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
            if result_file.exists():
                try:
                    res["jest_result"] = json.loads(result_file.read_text())
                except Exception:
                    res["result_file"] = str(result_file)
            return res
        except subprocess.TimeoutExpired:
            return {"error": "test run timed out"}

    def _commit_and_push(self, files, message="chore: add generated tests"):
        try:
            repo = Repo(self.repo_path)
            repo.index.add(files)
            commit = repo.index.commit(message)
            origin = repo.remote(name="origin")
            push_info = origin.push()
            return {"ok": True, "commit_hex": commit.hexsha, "push_info": [str(x) for x in push_info]}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def handle_request(self, feature: str, framework: str = "jest", commit: bool = False, run_tests: bool = True):
        """
        Main entry: produce test code from feature description, write file, optionally run and commit.
        """
        target_file = "src/LoginPage.js"  # simple heuristic for MVP
        raw = self._call_model(feature, target_file)
        parsed = self._extract_json(raw)

        if not parsed:
            # fallback: treat whole output as code and write to default file
            parsed = {"plan": "autogenerated", "filename": "tests/generated.test.js", "code": raw}

        filename = parsed.get("filename", "tests/generated.test.js")
        code = parsed.get("code", "")

        dest = self._write_test_file(filename, code)

        result = {"status": "written", "filename": str(dest.relative_to(self.repo_path)), "plan": parsed.get("plan", "")}

        # run tests
        if run_tests:
            result["test_run"] = self._run_tests()

        # commit & push
        if commit:
            result["commit"] = self._commit_and_push([str(dest.relative_to(self.repo_path))], message=f"chore: add tests for feature")

        return result
