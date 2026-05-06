# llm/inference.py

import ollama

MODEL_NAME = "tinyllama"


def build_prompt(query: str, context: str = "") -> str:
    if context:
        return (
            "You are an AI assistant in a distributed RAG system.\n"
            "Use the provided context to answer the user query.\n\n"
            f"Context:\n{context}\n\n"
            f"User query: {query}\n"
            "Answer:"
        )

    return (
        "You are an AI assistant.\n"
        f"User query: {query}\n"
        "Answer:"
    )


def run_llm(query: str, context: str = "", ollama_url: str = "http://localhost:11434") -> str:
    """
    Run real LLM inference using a local or remote Ollama server.
    """

    prompt = build_prompt(query, context)

    try:
        client = ollama.Client(host=ollama_url)

        response = client.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=False,
            options={
                "num_predict": 80
            }
        )

        return response["message"]["content"].strip()

    except Exception as e:
        raise RuntimeError(
            f"LLM inference failed on {ollama_url}: {str(e)}"
        )


def run_llm_batch(requests: list, ollama_url: str = "http://localhost:11434") -> list:
    results = []

    for query, context in requests:
        results.append(run_llm(query, context, ollama_url=ollama_url))

    return results