"""
The factory's services: the agents, the planner, and the webhooks into them (see README.md).
"""

import os
import re
from datetime import timedelta
from typing import Any

import restate
from pydantic import ValidationError
from restate import sessions
from restate.agents import Event, Tool, call, openai, pydantic_ai, subagent, webhook

from examples.factory import env, github, prompts
from examples.factory.answers import Issue, Plan
from examples.factory.signatures import CODER, REVIEWER, TRIAGE, by_an_agent

# Triage, the planner and the reviewer run on OpenAI's models ($RESTATE_AGENT_MODEL, else gpt-6-sol),
# and the coders on Anthropic's, unless CODER_MODEL says otherwise.
CODER_MODEL = os.environ.get("CODER_MODEL", "anthropic/claude-sonnet-5-5")


# --- The planner: an agent whose answer is a Plan ------------------------------------------------------

planner = openai.agent(
    "planner",
    description="Plans the work on an issue, from its title and description: a summary, and parts to build at once",
    system=prompts.PLANNER,
)


def _json_in(answer: str) -> str:
    """The JSON object in the model's answer, without the Markdown fence it may come in."""
    found = re.search(r"\{.*\}", answer, re.DOTALL)
    return found[0] if found else answer


@planner.handler
async def plan(ctx: sessions.SessionContext, issue: Issue) -> Plan:
    """Plan the work on an issue: the model's plan, checked against ``Plan``, or asked again if it doesn't fit.

    ``openai.prompt`` answers each prompt once, in the planner's conversation for the issue
    (``planner/ENG-1``): a retry gets the same answers, and doesn't ask the model again.
    """
    answer = await openai.prompt(ctx, f"# {issue.title}\n\n{issue.description}", system=prompts.PLANNER)
    for _ in range(3):
        try:
            return Plan.model_validate_json(_json_in(answer))
        except ValidationError as error:
            answer = await openai.prompt(
                ctx,
                f"That plan doesn't fit the schema: {error}\nAnswer again, with the JSON only.",
                system=prompts.PLANNER,
            )
    raise restate.TerminalError("The planner gave no plan that fits its schema")


# --- The agents -------------------------------------------------------------------------------------

reviewer = openai.agent(
    "reviewer",
    description="Reviews a pull request, as pushed, and posts its review on it: approved, or the changes to make",
    system=prompts.REVIEWER,
    tools=env.review_tools(REVIEWER),
)

CODER_TOOLS = [*env.CODING, env.pull_request_tool(CODER), env.pull_request_comment_tool(CODER)]


async def coder_tools(ctx: sessions.SessionContext) -> list[Tool]:
    return [*CODER_TOOLS, *await env.linear_tools(CODER)]


coder = pydantic_ai.agent(
    "coder",
    description="Builds a part of the plan, on a branch of its own, and opens its pull request; answers its reviews",
    model=CODER_MODEL,
    system=prompts.CODER,
    tools=coder_tools,
    inactivity_timeout=timedelta(minutes=10),  # a coder's commands (tests, builds) can take minutes
)


async def triage_tools(ctx: sessions.SessionContext) -> list[Tool]:
    return [
        *await env.linear_tools(TRIAGE),
        call(
            plan,
            description="Plan the work on an issue, from its title and description: a summary, and its parts",
            key=lambda ctx: ctx.key,  # the issue's planner: planner/ENG-1
        ),
        subagent(coder),
        subagent(reviewer),
    ]


triage = openai.agent("triage", system=prompts.TRIAGE, tools=triage_tools)


# --- The webhooks into them ------------------------------------------------------------------------


def issue_of(event: Event) -> str | None:
    """The issue a Linear event is for (its triage's key), or None to drop it.

    A new issue that is not a sub-issue, or a person's comment on one. The event's shape:
    https://linear.app/developers/webhooks
    """
    if event.get("action") != "create":
        return None
    data: dict[str, Any] = event.get("data") or {}
    if event.get("type") == "Issue" and not data.get("parentId"):
        return data.get("identifier")
    if event.get("type") == "Comment" and not by_an_agent(str(data.get("body") or "")):
        issue: dict[str, Any] = data.get("issue") or {}
        return issue.get("identifier")
    return None


linear_events = webhook(
    "linear",
    description="Linear's new issues, and comments on them, to triage",
    route=[triage],
    key=issue_of,
    secret=env.LINEAR_WEBHOOK_SECRET,
    signature_header="linear-signature" if env.LINEAR_WEBHOOK_SECRET else None,  # unsigned deliveries are taken
    idempotency_header="linear-delivery",
)

github_events = webhook(
    "github",
    description="GitHub's comments, reviews and failed CI runs on the coders' pull requests, to their coders",
    route=[coder],
    key=github.coder_of,
    message=github.what_happened,
    secret=env.GITHUB_WEBHOOK_SECRET,
    signature_header="x-hub-signature-256" if env.GITHUB_WEBHOOK_SECRET else None,
    idempotency_header="x-github-delivery",
)

app = restate.app([linear_events, github_events, triage, planner, coder, reviewer])
