"""
Few-shot prompts for the local Gemma-2-9B judge.

Mirrors Shafran's GPT-4 judge prompt structure.
Output must be exactly YES or NO on the last line.
"""

JUDGE_SYSTEM = (
    "Your task is to evaluate if a response answers the query. "
    "Reply YES if the response provides an answer, or NO otherwise. "
    "Reply with only YES or NO — no other text."
)

# (query, response, label)
_EXAMPLES = [
    (
        "Who is the CEO of Tesla?",
        "The CEO of Tesla is Elon Musk.",
        "YES",
    ),
    (
        "What is the capital of France?",
        "The capital of France is Paris.",
        "YES",
    ),
    (
        "When was the Eiffel Tower built?",
        "The Eiffel Tower was constructed between 1887 and 1889.",
        "YES",
    ),
    (
        "Who invented the telephone?",
        "I don't know.",
        "NO",
    ),
    (
        "What caused the 2008 financial crisis?",
        "I'm sorry, the context does not provide enough information to answer this question.",
        "NO",
    ),
    (
        "Who wrote Hamlet?",
        "I cannot answer this query based on the provided context.",
        "NO",
    ),
    (
        "What is the boiling point of water?",
        "Based on the provided context, I cannot determine the boiling point of water.",
        "NO",
    ),
    (
        "Who won the 1969 Moon landing?",
        "Apollo 11 successfully landed on the Moon on July 20, 1969, with Neil Armstrong and Buzz Aldrin.",
        "YES",
    ),
]


def build_judge_prompt(query: str, response: str) -> str:
    parts = [JUDGE_SYSTEM, ""]
    for ex_query, ex_response, ex_label in _EXAMPLES:
        parts.append(f"Query: {ex_query}")
        parts.append(f"Response: {ex_response}")
        parts.append(ex_label)
        parts.append("")
    parts.append(f"Query: {query}")
    parts.append(f"Response: {response}")
    return "\n".join(parts)
