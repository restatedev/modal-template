"""
The factory's services: the agents, the planner, and the webhooks into them (see README.md).
"""

import functools
import os
from datetime import timedelta
from typing import Any

import restate
from restate import sessions
from restate.agents import Event, call, openai, pydantic_ai, subagent, webhook

from examples.factory import env, github, prompts
from examples.factory.answers import Issue, Plan
from examples.factory.signatures import CODER, REVIEWER, TRIAGE, by_an_agent

# The models: OpenAI's for triage, the planner and the reviewer, and Anthropic's for the coders.
OPENAI_MODEL = os.environ.get("RESTATE_AGENT_MODEL", "openai/gpt-6-sol")
CODER_MODEL = os.environ.get("CODER_MODEL", "anthropic/claude-sonnet-5-5")


# --- The planner: a plain service, asked once, its answer a Plan --------------------------------------


@functools.cache
def openai_client() -> Any:
    from openai import AsyncOpenAI  # pylint: disable=import-outside-toplevel

    return AsyncOpenAI()


planner = restate.Service(
    "planner",
    description="Plans the work on an issue, from its title and description: a summary, and parts to build at once",
)


@planner.handler()
async def plan(ctx: restate.Context, issue: Issue) -> Plan:
    """Plan the work on an issue: the model's answer checked against ``Plan``, and journaled, so it is asked once."""

    async def ask() -> Plan:
        response = await openai_client().responses.parse(
            model=OPENAI_MODEL.removeprefix("openai/"),
            instructions=prompts.PLANNER,
            input=f"# {issue.title}\n\n{issue.description}",
            text_format=Plan,
            store=False,
        )
        if response.output_parsed is None:
            raise restate.TerminalError("The model gave no plan")
        return response.output_parsed

    return await ctx.run_typed("ask the model for a plan", ask)


# --- The agents -------------------------------------------------------------------------------------

reviewer = openai.agent(
    "reviewer",
    description="Reviews a pull request, as pushed, and posts its review on it: approved, or the changes to make",
    model=OPENAI_MODEL,
    system=prompts.REVIEWER,
    tools=env.review_tools(REVIEWER),
)

CODER_TOOLS = [*env.CODING, env.pull_request_tool(CODER), env.pull_request_comment_tool(CODER)]


async def coder_tools(ctx: sessions.SessionContext) -> list[Any]:
    return [*CODER_TOOLS, *await env.linear_tools(CODER)]


coder = pydantic_ai.agent(
    "coder",
    description="Builds a part of the plan, on a branch of its own, and opens its pull request; answers its reviews",
    model=CODER_MODEL,
    system=prompts.CODER,
    tools=coder_tools,
    inactivity_timeout=timedelta(minutes=10),  # a coder's commands (tests, builds) can take minutes
)


async def triage_tools(ctx: sessions.SessionContext) -> list[Any]:
    return [
        *await env.linear_tools(TRIAGE),
        call(plan, description="Plan the work on an issue, from its title and description: a summary, and its parts"),
        subagent(coder),
        subagent(reviewer),
    ]


triage = openai.agent("triage", model=OPENAI_MODEL, system=prompts.TRIAGE, tools=triage_tools)


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
