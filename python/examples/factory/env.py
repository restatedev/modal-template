"""
The factory's environment, and the agents' tools on it:

- Linear, with LINEAR_API_KEY (and LINEAR_WEBHOOK_SECRET, its webhook's signing secret).
- The project on GitHub, with GITHUB_REPO (owner/name) and GITHUB_TOKEN (and
  GITHUB_WEBHOOK_SECRET, its webhook's secret).
- WORKSPACE, where the coders work (/tmp/restate-factory by default).

LINEAR_MCP_URL, GITHUB_MCP_URL and GITHUB_REMOTE point it elsewhere, such as to
GitHub Enterprise; they default to Linear's and GitHub's.
"""

import asyncio
import os
from collections.abc import Awaitable
from pathlib import Path
from typing import Annotated, Any

import mcp_types
import restate
from pydantic import Field
from restate import sessions
from restate.agents import Tool, ToolCall, tool

from examples.factory.github import pull_request_mark
from examples.factory.mcp_client import McpClient
from examples.factory.signatures import signer
from examples.factory.workspace import GitError, Pushed, clone, project, pushed


def required(name: str, what: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"The factory needs {name}: {what} (see examples/factory/README.md)")
    return value


LINEAR_API_KEY = required("LINEAR_API_KEY", "a Linear API key it acts with")
LINEAR_WEBHOOK_SECRET = os.environ.get("LINEAR_WEBHOOK_SECRET")
GITHUB_REPO = required("GITHUB_REPO", "the GitHub repository it builds in, as owner/name")
GITHUB_TOKEN = required("GITHUB_TOKEN", f"a token that can push to {GITHUB_REPO}, and open its pull requests")
GITHUB_WEBHOOK_SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET")
WORKSPACE = os.environ.get("WORKSPACE", "/tmp/restate-factory")

linear = McpClient(
    os.environ.get("LINEAR_MCP_URL", "https://mcp.linear.app/mcp"), {"authorization": f"Bearer {LINEAR_API_KEY}"}
)
github = McpClient(
    os.environ.get("GITHUB_MCP_URL", "https://api.githubcopilot.com/mcp/"), {"authorization": f"Bearer {GITHUB_TOKEN}"}
)
REMOTE = os.environ.get("GITHUB_REMOTE", f"https://github.com/{GITHUB_REPO}.git")
OWNER, REPO = GITHUB_REPO.split("/", 1)

_project: str | None = None
_project_lock = asyncio.Lock()


async def remote() -> str:
    """The project's repository, with a first commit if it is empty: made ready once, when first needed.

    One at a time: two coders starting at once on an empty repository would both push its first commit.
    """
    global _project  # a cache: made again next time if it failed
    async with _project_lock:
        if _project is None:
            _project = await project(REMOTE)
    return _project


# --- Linear's tools, signed -----------------------------------------------------------------------


def _writing(listed: mcp_types.Tool, arguments: dict[str, Any]) -> str | None:
    """The field of a Linear tool's input that an agent writes, to sign.

    A comment's ``body``, or a new issue's ``description`` (an issue with an ``id`` is updated: it is signed
    already). By the input, not the tool's name alone: Linear names its tools as it likes.
    """
    fields: dict[str, Any] = listed.input_schema.get("properties") or {}
    if "comment" in listed.name and "body" in fields and isinstance(arguments.get("body"), str):
        return "body"
    if (
        "issue" in listed.name
        and "description" in fields
        and isinstance(arguments.get("description"), str)
        and "id" not in arguments
    ):
        return "description"
    return None


async def linear_tools(who: str) -> list[Tool]:
    """Linear's tools for the agent ``who``: what it writes opens with its name, and its part.

    The agents all write as one Linear user: this tells them apart. Asked of Linear's MCP server when an
    agent first needs them.
    """
    sign = signer(who)

    def signed(listed: mcp_types.Tool, arguments: dict[str, Any], call: ToolCall) -> dict[str, Any]:
        field = _writing(listed, arguments)
        return arguments if field is None else {**arguments, field: sign(call.ctx.key, str(arguments[field]))}

    return await linear.tools(signed)


# --- GitHub: the pull requests, signed ------------------------------------------------------------


def pull_request_tool(who: str) -> Tool:
    """``create_pull_request`` for ``who``: its description opens with who opened it, and ends with the coder's mark."""
    sign = signer(who)

    @tool(name="create_pull_request")
    async def create_pull_request(
        ctx: sessions.SessionContext,
        title: str,
        body: Annotated[str, Field(description="What it does, and how it was tested, in Markdown")],
        head: Annotated[str, Field(description="The branch with the changes, pushed already")],
        base: Annotated[str, Field(description="The branch it goes into")],
    ) -> str:
        """Open a pull request of the project: from the branch ``head`` into ``base``. Answers with its link."""
        described = f"{sign(ctx.key, body)}\n\n{pull_request_mark(ctx.key)}"
        arguments = {"owner": OWNER, "repo": REPO, "title": title, "body": described, "head": head, "base": base}
        return await github.call("create_pull_request", arguments)

    return create_pull_request


def pull_request_comment_tool(who: str) -> Tool:
    """``comment_on_pull_request`` for ``who``: a comment on a pull request, signed, such as a review or an answer to one."""
    sign = signer(who)

    @tool(name="comment_on_pull_request")
    async def comment_on_pull_request(
        ctx: sessions.SessionContext,
        number: Annotated[int, Field(description="The pull request's number")],
        body: Annotated[str, Field(description="The comment, in Markdown")],
    ) -> str:
        """Comment on a pull request: a review, or an answer to one."""
        arguments = {"owner": OWNER, "repo": REPO, "issue_number": number, "body": sign(ctx.key, body)}
        return await github.call("add_issue_comment", arguments)

    return comment_on_pull_request


# --- The coder's tools: its clone of the project ---------------------------------------------------

# The keys a coder's commands don't see. GITHUB_TOKEN stays: `git push` reads it (see workspace.py).
HIDDEN = ("LINEAR_API_KEY", "LINEAR_WEBHOOK_SECRET", "GITHUB_WEBHOOK_SECRET", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
HIDDEN_PREFIXES = ("RESTATE_",)
OUTPUT = 20_000  # characters of a command's output the model sees: the end of it


async def folder_of(ctx: sessions.SessionContext) -> Path:
    """The coder's clone of the project, on its own branch: made the first time it needs it."""
    return await clone(WORKSPACE, await remote(), ctx.key)


def _inside(folder: Path, path: str) -> Path:
    target = (folder / path).resolve()
    if not target.is_relative_to(folder):
        raise restate.TerminalError(f"{path} is outside your clone of the project")
    return target


@tool
async def bash(
    ctx: sessions.SessionContext,
    command: Annotated[str, Field(description="The command, run with bash in your clone of the project")],
) -> str:
    """Run a command in your clone of the project (5 minutes at most), and get its exit code and output."""
    folder = await folder_of(ctx)
    env = {k: v for k, v in os.environ.items() if k not in HIDDEN and not k.startswith(HIDDEN_PREFIXES)}
    process = await asyncio.create_subprocess_exec(
        "bash", "-c", command, cwd=folder, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    try:
        out, _ = await asyncio.wait_for(process.communicate(), timeout=300)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise restate.TerminalError("The command took more than 5 minutes, and was stopped") from None
    text = out.decode(errors="replace")
    return f"exit code {process.returncode}\n{text if len(text) <= OUTPUT else '…' + text[-OUTPUT:]}"


@tool(name="read_file")
async def read_own_file(ctx: sessions.SessionContext, path: str) -> str:
    """A file of your clone of the project, by its path in it."""
    target = _inside(await folder_of(ctx), path)
    if not target.is_file():
        raise restate.TerminalError(f"There is no file {path}")
    return target.read_text(errors="replace")


@tool
async def write_file(ctx: sessions.SessionContext, path: str, content: str) -> str:
    """Write a file of your clone of the project, by its path in it, all of it: made, or replaced."""
    target = _inside(await folder_of(ctx), path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Wrote {path}"


CODING = [bash, read_own_file, write_file]


# --- The reviewer's tools: the pull request it reviews, as pushed ------------------------------------


async def pull_request_of(ctx: sessions.SessionContext) -> Pushed:
    """The pull request a reviewer reviews: its coder's, as pushed (``reviewer/ENG-1.parser`` reviews
    ``coder/ENG-1.parser``'s), read in a clone of the project, made here if this machine has none."""
    await folder_of(ctx)
    return pushed(WORKSPACE, ctx.key)


async def _reading(what: str, read: Awaitable[str]) -> str:
    try:
        return await read
    except GitError as error:
        return f"Could not read {what} (pushed yet?): {error}"


@tool
async def pull_request_diff(ctx: sessions.SessionContext) -> str:
    """The pull request's changes, as pushed: the files it changes, and their diff against main."""
    return await _reading("the changes", (await pull_request_of(ctx)).changes())


@tool(name="read_file")
async def read_pushed_file(ctx: sessions.SessionContext, path: str) -> str:
    """A file of the pull request, as pushed, by its path in the project."""
    return await _reading(path, (await pull_request_of(ctx)).file(path))


def review_tools(who: str) -> list[Tool]:
    return [pull_request_diff, read_pushed_file, pull_request_comment_tool(who)]
