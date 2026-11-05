"""
An agent that improves its own harness

An ``AgentObject`` with an agent loop of your own: no framework's ``agent()``,
the model asked explicitly (OpenAI's Responses API), and the loop written out.
That is how you take full control: what the model sees each turn, which tools
it has, and what is stored.

Here the harness, the agent's instructions, is not in the code but in each
user's database, one version after another. Each turn reads the latest. And
``reflect`` asks the model to look at the conversation, find what the user
corrected or asked for more than once, and rewrite the harness to match, with
the reason: the agent changes itself, per user, and each change is kept, with
a diff against the version before, so you can see what it learned, and roll it
back::

    uv run python -m restate serve examples.self_improving:app
    curl localhost:8080/learner/alice/chat -d 'What should I pack for a weekend in Lisbon?'
    curl localhost:8080/learner/alice/chat -d 'Way too long. Three bullets at most, please.'
    curl localhost:8080/learner/alice/reflect        # a new version of the harness, and why
    curl localhost:8080/learner/alice/harness        # every version, with its diff
    curl localhost:8080/learner/alice/chat -d 'And for Oslo in winter?'   # answered by the new version
    curl localhost:8080/learner/alice/rollback --json 1                  # back to the first, as a new version

Or chat from a terminal: ``uv run python -m restate.agents chat learner/alice``.

What Restate does for the loop, as in any agent: each of the model's replies is
stored before its tools run, and each tool result as it comes, so a crash
resumes the turn without asking the model again for what it already answered.
``reflect`` is a handler of your own, so it is yours to make safe to run twice:
it stores its version under the request's id, and a retry returns that version
instead of asking the model again.
"""

import asyncio
import difflib
import functools
import itertools
import json
import os
from collections.abc import Awaitable, Iterable, Sequence
from typing import Any

import restate
from pydantic import BaseModel
from restate import sessions
from restate.agents import AgentObject, ToolCall, tool
from restate.agents.internal import Call, Conversation, Parsed, Result, answer_once, run_tool

MODEL = os.environ.get("RESTATE_AGENT_MODEL", "openai/gpt-6-sol").removeprefix("openai/")

FIRST_HARNESS = "You are a helpful travel assistant. Answer the user's questions about trips."

REFLECT = """You improve the instructions (the harness) of an assistant, from its conversation with one user.
Find what the user corrected, asked for, or complained about, that should hold for every later answer:
length, format, tone, what to include or leave out. Rewrite the harness to include it, keeping what still
holds. Change nothing the conversation gives no reason to change. If there is nothing to learn, return
the harness as it is."""


# --- The tools --------------------------------------------------------------------------------------


@tool
def get_weather(city: str) -> str:
    """The current weather in a city."""
    return f"Sunny and 21°C in {city}"  # a stand-in for a real weather API


TOOLS = {t.name: t for t in (get_weather,)}
SCHEMAS = [
    {"type": "function", "name": t.name, "description": t.description, "parameters": t.input_schema, "strict": False}
    for t in TOOLS.values()
]


# --- The harness: versions in the user's database ---------------------------------------------------


class Version(BaseModel):
    version: int
    system: str
    why: str
    diff: str = ""  # against the version before


class History(BaseModel):
    versions: list[Version]


async def harness_of(db: sessions.DatabaseReader) -> Version:
    """The latest version of the harness, or the first one if the agent never changed it."""
    row = await db.get("SELECT version, system, why FROM harness ORDER BY version DESC LIMIT 1")
    if row is None:
        return Version(version=1, system=FIRST_HARNESS, why="The first version.")
    return Version(version=int(row["version"] or 0), system=str(row["system"]), why=str(row["why"]))


async def save(ctx: sessions.SessionContext, system: str, why: str) -> Version:
    """Store ``system`` as the next version, once per request: a retry finds the version it stored."""
    async with ctx.db.transaction() as tx:
        if await tx.get("SELECT 1 FROM harness") is None:  # keep the first version, to diff and roll back to
            await tx.execute(
                "INSERT INTO harness (version, system, why) VALUES (1, ?, ?)", [FIRST_HARNESS, "The first version."]
            )
        latest = await harness_of(tx)
        await tx.execute(
            "INSERT OR IGNORE INTO harness (version, system, why, request) VALUES (?, ?, ?, ?)",
            [latest.version + 1, system, why, ctx.request.id],
        )
        row = await tx.get("SELECT version, system, why FROM harness WHERE request = ?", [ctx.request.id])
        assert row is not None  # inserted now, or by an earlier attempt of this request
        return Version(version=int(row["version"] or 0), system=str(row["system"]), why=str(row["why"]))


# --- The model, asked explicitly ---------------------------------------------------------------------


@functools.cache
def openai_client() -> Any:
    import openai  # pylint: disable=import-outside-toplevel

    return openai.AsyncOpenAI()


async def complete(system: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ask the model, with this turn's harness, and return its reply's items (text, tool calls, reasoning)."""
    response = await openai_client().responses.create(
        model=MODEL,
        instructions=system,
        input=items,
        tools=SCHEMAS,
        # The conversation is in Restate, not OpenAI's: each reasoning item carries its own (encrypted) content.
        store=False,
        include=["reasoning.encrypted_content"],
    )
    return [item.model_dump(exclude_none=True, by_alias=True) for item in response.output]


class Revision(BaseModel):
    system: str
    """The harness, rewritten."""
    why: str
    """What in the conversation made it change, in a sentence or two."""


async def revise(harness: str, transcript: str) -> Revision:
    """Ask the model for a better harness, from the conversation: one call, its answer checked against ``Revision``."""
    response = await openai_client().responses.parse(
        model=MODEL,
        instructions=REFLECT,
        input=f"<harness>\n{harness}\n</harness>\n\n<conversation>\n{transcript}\n</conversation>",
        text_format=Revision,
        store=False,
    )
    revision = response.output_parsed
    if revision is None:
        raise restate.TerminalError("The model gave no revision")
    return revision


# --- The conversation's items: the Responses API's ------------------------------------------------


class Messages:
    """How the conversation stores, reads and makes the loop's items (``restate.agents.internal.Items``)."""

    def dump(self, item: dict[str, Any]) -> str:
        return json.dumps(item)

    def load(self, stored: str) -> dict[str, Any]:
        return json.loads(stored)

    def prompt(self, text: str) -> dict[str, Any]:
        return {"role": "user", "content": text}

    def reply(self, text: str) -> dict[str, Any]:
        return {"role": "assistant", "content": text}

    def results(self, results: Sequence[tuple[Call, str]]) -> list[dict[str, Any]]:
        return [result(call.id, text) for call, text in results]

    def read(self, item: dict[str, Any]) -> Parsed:
        match item.get("type", "message"):
            case "message":
                return Parsed((item["role"], _text(item["content"])))
            case "function_call":
                return Parsed(calls=[Call(item["call_id"], item["name"], _arguments(item["arguments"]))])
            case "function_call_output":
                return Parsed(results=[Result(item["call_id"], item["output"])])
            case _:  # the model's reasoning, say
                return Parsed()


def result(call_id: str, output: str) -> dict[str, Any]:
    """The item that gives tool call ``call_id`` its output."""
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _text(content: str | list[dict[str, Any]]) -> str:
    """The text of a message: a string, or its parts' (the model's ``output_text``)."""
    return content if isinstance(content, str) else "".join(str(part.get("text", "")) for part in content)


def _arguments(arguments: str) -> Any:
    try:
        return json.loads(arguments)
    except ValueError:
        return {}  # the tool then tells the model its arguments do not fit


MESSAGES = Messages()


async def all_of(runs: Iterable[Awaitable[None]]) -> None:
    """Run ``runs`` at once; if one fails, cancel the others, and fail with its error once they have unwound."""
    tasks = [asyncio.ensure_future(r) for r in runs]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)  # let them unwind, before the turn closes
        raise


# --- The loop ---------------------------------------------------------------------------------------


async def answer(ctx: sessions.SessionContext, text: str) -> str:
    """Answer a message with the latest harness: the model, and the tools it calls, until it answers."""
    system = (await harness_of(ctx.db)).system  # read each turn: the harness changes as the agent learns

    async def run(conversation: Conversation) -> str:
        async def run_call(call: Call) -> None:
            found = TOOLS.get(call.name)
            output = (
                await run_tool(found, call.input, ToolCall(ctx, call.id))
                if found is not None
                else f"Error: there is no tool named {call.name}"
            )
            await conversation.append(result(call.id, output))  # stored as soon as it comes

        if (final := conversation.final_reply()) is not None:
            return final  # a crash came after the final reply was stored, before its answer was
        while True:
            # The tool calls with no result yet (the last reply's, or those a crash cut off), all at once.
            await all_of(run_call(call) for call in conversation.unanswered())
            reply = await complete(system, [e.item for e in conversation.entries])
            await conversation.append_reply(*reply)
            if not any(item.get("type") == "function_call" for item in reply):
                return conversation.final_reply() or ""

    return await answer_once(ctx, text, MESSAGES, run)


learner = AgentObject(
    "learner",
    # The harness's versions, in each user's database next to the conversation.
    migrations=[
        "CREATE TABLE harness (version INTEGER PRIMARY KEY, system TEXT NOT NULL, why TEXT NOT NULL, request TEXT UNIQUE)"
    ],
)
learner.chat(answer, items=MESSAGES)


@learner.handler
async def reflect(ctx: sessions.SessionContext) -> Version:
    """Look at the conversation, and store a better harness: the agent improving itself."""
    stored = await ctx.db.get("SELECT version, system, why FROM harness WHERE request = ?", [ctx.request.id])
    if stored is not None:  # a retry: this request already stored its version, so don't ask the model again
        return Version(version=int(stored["version"] or 0), system=str(stored["system"]), why=str(stored["why"]))
    conversation = await Conversation.load(ctx.db, MESSAGES)
    transcript = "\n".join(f"{e.said[0]}: {e.said[1]}" for e in conversation.entries if e.said is not None)
    current = await harness_of(ctx.db)
    if not transcript:
        return current
    revision = await revise(current.system, transcript)
    if revision.system.strip() == current.system.strip():
        return current  # nothing to learn
    return await save(ctx, revision.system, revision.why)


@learner.handler
async def rollback(ctx: sessions.SessionContext, version: int) -> Version:
    """Go back to an earlier version of the harness, as a new version: the history keeps every change."""
    row = await ctx.db.get("SELECT system FROM harness WHERE version = ?", [version])
    system = FIRST_HARNESS if row is None and version == 1 else None if row is None else str(row["system"])
    if system is None:
        raise restate.TerminalError(f"There is no version {version}")
    return await save(ctx, system, f"Rolled back to version {version}.")


@learner.view
async def harness(ctx: sessions.ViewContext) -> History:
    """Every version of the harness, oldest first, each with its diff against the one before."""
    rows = await ctx.db.query("SELECT version, system, why FROM harness ORDER BY version")
    versions = [Version(version=int(r["version"] or 0), system=str(r["system"]), why=str(r["why"])) for r in rows]
    if not versions:
        return History(versions=[await harness_of(ctx.db)])
    for before, after in itertools.pairwise(versions):
        after.diff = "\n".join(
            difflib.unified_diff(
                before.system.splitlines(),
                after.system.splitlines(),
                f"v{before.version}",
                f"v{after.version}",
                lineterm="",
            )
        )
    return History(versions=versions)


app = restate.app([learner])
