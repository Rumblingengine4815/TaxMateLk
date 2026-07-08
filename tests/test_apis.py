

import os
import sys
import argparse
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()


def suggest_install(package_name, extra=""):
    console.print(f"  [yellow]Hint:[/yellow] Install the package: [bold]pip install {package_name} {extra}[/bold]")
    console.print("  Or create/refresh a requirements file: [bold]pip freeze > requirements.txt[/bold]")
    console.print("  To run the app with a modern ASGI runner: [bold]uvicorn app:app --reload --host 0.0.0.0 --port 8000[/bold]")


def test_groq():
    """Test Groq API — your primary free LLM."""
    console.print("\n[bold cyan]Testing Groq API...[/bold cyan]")
    try:
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": "Reply with exactly: GROQ_OK"
            }],
            max_tokens=10
        )
        reply = response.choices[0].message.content.strip()
        assert "GROQ_OK" in reply or len(reply) > 0
        console.print(f"  [green]✅ Groq working[/green] — model: llama-3.1-8b-instant")
        console.print(f"     Reply: {reply}")
        console.print(f"     Tokens used: {response.usage.total_tokens}")
        return True
    except KeyError:
        console.print("  [red]❌ GROQ_API_KEY not set in .env[/red]")
        return False
    except ImportError as e:
        console.print(f"  [red]❌ Groq client library not installed: {e}[/red]")
        suggest_install("groq")
        return False
    except Exception as e:
        console.print(f"  [red]❌ Groq failed: {e}[/red]")
        return False


def test_gemini():
    """Test Gemini API — your fallback + embeddings."""
    console.print("\n[bold cyan]Testing Gemini API...[/bold cyan]")
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            console.print("  [red]❌ GEMINI_API_KEY not set in .env[/red]")
            return False

        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents="Reply with exactly: GEMINI_OK",
            )
            reply = (response.text or "").strip()
            console.print("  [green]✅ Gemini working[/green] — model: gemini-2.0-flash")
            console.print(f"     Reply: {reply}")
            return True
        except Exception as primary_error:
            console.print(f"  [yellow]Primary Gemini call failed:[/yellow] {primary_error}")
            console.print("  [yellow]Trying fallback model gemini-2.5-flash...[/yellow]")
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents="Reply with exactly: GEMINI_OK",
                )
                reply = (response.text or "").strip()
                console.print("  [green]✅ Gemini working[/green] — model: gemini-2.5-flash")
                console.print(f"     Reply: {reply}")
                return True
            except Exception as secondary_error:
                console.print(f"  [red]❌ Gemini failed: {secondary_error}[/red]")
                console.print("  [yellow]Hint:[/yellow] Install or upgrade the current SDK: [bold]pip install --upgrade google-genai[/bold]")
                console.print("  [yellow]Hint:[/yellow] Your key can be brand new; this error usually means the SDK or model name is outdated or unavailable on your account.")
                return False
    except ImportError as e:
        console.print(f"  [red]❌ Gemini client library not installed: {e}[/red]")
        suggest_install("google-genai")
        return False
    except Exception as e:
        console.print(f"  [red]❌ Gemini failed: {e}[/red]")
        return False


def test_huggingface():
    """Test HuggingFace token — needed for fine-tuning and deployment."""
    console.print("\n[bold cyan]Testing HuggingFace token...[/bold cyan]")
    try:
        from huggingface_hub import whoami, login
        # Accept multiple env var names for the HuggingFace token
        token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY") or os.environ.get("HF_API") or os.environ.get("HF_KEY")
        if not token:
            console.print("  [red]❌ HuggingFace token not set in .env (set HF_TOKEN or HF_API_KEY)[/red]")
            return False
        login(token=token, add_to_git_credential=False)
        info = whoami()
        console.print(f"  [green]✅ HuggingFace working[/green]")
        console.print(f"     Logged in as: {info.get('name') or info.get('user', {}).get('name')}")
        return True
    except ImportError as e:
        console.print(f"  [red]❌ HuggingFace client library not installed: {e}[/red]")
        suggest_install("huggingface-hub")
        return False
    except Exception as e:
        console.print(f"  [red]❌ HuggingFace failed: {e}[/red]")
        return False


def test_chromadb():
    """Test ChromaDB works locally — no API key needed."""
    console.print("\n[bold cyan]Testing ChromaDB (local)...[/bold cyan]")
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        client = chromadb.Client()
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        col = client.create_collection("test", embedding_function=ef)
        col.add(
            documents=["WHT applies to photographers from June 2026"],
            ids=["test_1"]
        )
        results = col.query(query_texts=["photographer tax"], n_results=1)
        assert len(results["documents"][0]) > 0
        console.print(f"[green]✅ ChromaDB working[/green] — local, no API key needed")
        console.print(f"Test retrieval: '{results['documents'][0][0][:60]}...'")
        return True
    except Exception as e:
        console.print(f"[red]❌ ChromaDB failed: {e}[/red]")
        return False
    except ImportError as e:
        console.print(f"[red]❌ ChromaDB client library not installed: {e}[/red]")
        suggest_install("chromadb sentence-transformers")
        return False


def main():
    console.print(Panel.fit(
        "[bold]TaxMate LK — API Connection Tests[/bold]\n"
        "Run this before writing any other code.",
        style="blue"
    ))

    parser = argparse.ArgumentParser(description="Run API connectivity checks")
    parser.add_argument("--only", "-o", nargs="*",
                        help="Run only the named checks (Groq, Gemini, HuggingFace, ChromaDB)")
    args = parser.parse_args()

    available = {
        "Groq": test_groq,
        "Gemini": test_gemini,
        "HuggingFace": test_huggingface,
        "ChromaDB": test_chromadb,
    }

    if args.only:
        names = [n for n in args.only]
    else:
        names = list(available.keys())

    results = {}
    for name in names:
        fn = available.get(name)
        if not fn:
            console.print(f"[yellow]Warning:[/yellow] Unknown check '{name}' — skipping")
            results[name] = False
            continue
        results[name] = fn()

    console.print("\n" + "="*50)
    console.print("[bold]SUMMARY[/bold]")
    all_passed = True
    for name, passed in results.items():
        icon = "[green]✅[/green]" if passed else "[red]❌[/red]"
        console.print(f"  {icon} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        console.print("\n[bold green]All APIs working. You're ready to build.[/bold green]")
        console.print("Next step: uv run python src/ingest.py")
    else:
        console.print("\n[bold red]Fix the failing APIs before continuing.[/bold red]")
        console.print("Check your .env file — copy from .env.example and add real keys.")
        sys.exit(1)


if __name__ == "__main__":
    main()
