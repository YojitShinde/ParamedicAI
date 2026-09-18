"""
ParamedicAI CLI
Interactive command-line interface for testing the RAG bot.

Usage:
    python main.py                                  # Interactive chat mode
    python main.py --query "describe the emergency"  # Single query mode
"""

import argparse
import sys

from rag_engine import ParamedicAI


def print_banner():
    """Print the ParamedicAI banner."""
    banner = """
+============================================================+
|                      PARAMEDIC AI                          |
|                                                            |
|                                                            |
|  [!] FOR EDUCATIONAL USE ONLY -- NOT A SUBSTITUTE FOR      |
|      PROFESSIONAL MEDICAL ADVICE. CONTACT LOCAL EMERGENCY  |
|      SERVICES FOR URGENT HELP.                             |
+============================================================+
"""
    print(banner)


def single_query(bot, query):
    """Run a single query and print the response."""
    print(f"\n{'=' * 60}")
    print(f"Query: {query}")
    print(f"{'=' * 60}\n")

    result = bot.query(query)

    print(result["response"])
    print(f"\n{'=' * 60}")
    print(f"[Sources consulted]: {', '.join(result['sources'])}")
    print(f"{'=' * 60}")


def interactive_mode(bot):
    """Run an interactive chat loop."""
    print("Type your emergency scenario below. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting ParamedicAI. Stay safe!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\nExiting ParamedicAI. Stay safe!")
            break

        print(f"\nParamedicAI:\n")

        result = bot.query(user_input)
        print(result["response"])

        print(f"\n[Sources]: {', '.join(result['sources'])}")
        print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="ParamedicAI"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single query mode: provide your emergency scenario as a string",
    )
    args = parser.parse_args()

    print_banner()

    # Initialize the bot
    bot = ParamedicAI()

    if args.query:
        single_query(bot, args.query)
    else:
        interactive_mode(bot)


if __name__ == "__main__":
    main()
