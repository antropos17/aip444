#!/usr/bin/env python3
"""
pr-advice - CLI tool: explains a GitHub PR (diff + comments) using an LLM.
Lab 3 / AIP444.

Usage:
    python pr-advice.py https://github.com/microsoft/vscode/pull/289801

Before running:
    pip install requests
    set OPENROUTER_API_KEY=sk-or-...        (Windows CMD)
    $env:OPENROUTER_API_KEY="sk-or-..."     (Windows PowerShell)
    export OPENROUTER_API_KEY=sk-or-...     (Linux/Mac)
"""

import os
import re
import sys
import json
import requests
from typing import List, Dict, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Auto-router that picks an available free model. Alternatives if needed:
#   meta-llama/llama-3.3-70b-instruct:free
#   deepseek/deepseek-r1-distill:free
MODEL = "openrouter/free"
MAX_DIFF_CHARS = 95_000
USER_AGENT = "AIP444-Lab-03"


# ---------------------------------------------------------------------------
# Step 1: Parse the URL
# ---------------------------------------------------------------------------
def parse_pr_url(url: str) -> Tuple[str, str, int]:
    """Return (owner, repo, pr_number) or exit with an error."""
    pattern = r"^https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)"
    match = re.match(pattern, url.strip())
    if not match:
        print("ERROR: this is not a valid GitHub pull request URL.")
        print("Example: https://github.com/microsoft/vscode/pull/289801")
        sys.exit(1)

    owner, repo, number = match.group(1), match.group(2), int(match.group(3))
    return owner, repo, number


# ---------------------------------------------------------------------------
# Step 2: Fetch the DIFF
# ---------------------------------------------------------------------------
def fetch_diff(owner: str, repo: str, number: int) -> str:
    """Fetch the .diff. Truncate and warn if it exceeds the limit."""
    url = f"https://github.com/{owner}/{repo}/pull/{number}.diff"
    resp = requests.get(url, headers={"User-Agent": USER_AGENT})
    if resp.status_code != 200:
        raise Exception(f"Failed to fetch diff: {resp.status_code}")

    diff = resp.text
    if len(diff) > MAX_DIFF_CHARS:
        print(f"WARNING: diff truncated ({len(diff)} -> {MAX_DIFF_CHARS} chars).")
        diff = diff[:MAX_DIFF_CHARS] + "\n\n...[Diff Truncated]..."
    return diff


# ---------------------------------------------------------------------------
# Step 3: Fetch the comments via the GitHub API
# ---------------------------------------------------------------------------
def fetch_comments(owner: str, repo: str, issue_num: int) -> List[Dict[str, str]]:
    """Return a list of {username, body, date}."""
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_num}/comments"
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        raise Exception(
            f"GitHub API Error: {resp.status_code} "
            f"(without a token the limit is 60 requests/hour - on 403 wait an hour)"
        )

    return [
        {
            "username": item["user"]["login"],
            "body": item["body"] or "",
            "date": item["updated_at"],
        }
        for item in resp.json()
    ]


# ---------------------------------------------------------------------------
# Step 4: Build the prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a Senior Software Engineer mentoring a junior developer. Your tone is \
educational but rigorous. You value code safety, correctness, and long-term \
maintainability over cleverness. You explain trade-offs clearly and never \
flatter; you point out real risks honestly.

You will receive two inputs:
1. A unified DIFF inside a fenced ```diff code block - the technical reality of the change.
2. A conversation thread inside <thread> XML tags, with each message as \
<comment username="..." date="...">...</comment> - the human context.

Follow this reasoning process before writing anything:
1. First, read the diff carefully to understand exactly what changed technically.
2. Next, read the <thread> to understand the human concerns, disagreements, and decisions.
3. Next, reflect on underlying assumptions, constraints, edge cases, and how this fits the larger codebase.
4. Finally, synthesize all of the above into the report.

Output ONLY a Markdown report with these exact sections and headings:

## tl;dr
A single sentence (max 30 words) summarizing the PR's purpose.

## Stakeholders
A bulleted list of every person who participated, each with a one-line description of their stance or contribution.

## Changes
A file-by-file breakdown of what changed and why, written so a junior developer can follow it.

## Risks
Potential bugs, unhandled edge cases, or hidden assumptions. Rate each as **Low**, **Medium**, or **High** severity.

## Learning
Exactly 3 Socratic questions that test the junior's understanding of these specific changes \
(e.g. reference real lines, functions, or decisions from the diff).
"""


def build_user_message(diff: str, comments: List[Dict[str, str]]) -> str:
    """Assemble the user message with proper delimiters."""
    parts = [
        "Here is the pull request to analyze.\n",
        "### Code changes (diff):",
        f"```diff\n{diff}\n```\n",
        "### Conversation:",
    ]

    if comments:
        parts.append("<thread>")
        for c in comments:
            parts.append(
                f'<comment username="{c["username"]}" date="{c["date"]}">\n'
                f'{c["body"]}\n</comment>'
            )
        parts.append("</thread>")
    else:
        parts.append("<thread>(No comments on this PR.)</thread>")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Step 5: Call OpenRouter
# ---------------------------------------------------------------------------
def call_llm(system_prompt: str, user_message: str) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: environment variable OPENROUTER_API_KEY is not set.")
        sys.exit(1)

    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }),
    )
    if resp.status_code != 200:
        raise Exception(f"OpenRouter Error: {resp.status_code} - {resp.text}")

    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python pr-advice.py <github-pr-url>")
        sys.exit(1)

    url = sys.argv[1]
    owner, repo, number = parse_pr_url(url)
    print(f"Analyzing PR: {owner}/{repo} #{number}\n")

    print("Fetching diff...")
    diff = fetch_diff(owner, repo, number)

    print("Fetching comments...")
    comments = fetch_comments(owner, repo, number)
    print(f"Comments found: {len(comments)}\n")

    user_message = build_user_message(diff, comments)

    print("Sending to LLM...\n")
    report = call_llm(SYSTEM_PROMPT, user_message)

    print("=" * 70)
    print(report)
    print("=" * 70)


if __name__ == "__main__":
    main()
