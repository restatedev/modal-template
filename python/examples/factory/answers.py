"""
The planner's answer, structured: a ``Plan``, and how to ask a model for one.
The model replies with JSON of its schema (the planner's prompt has it);
``ask_for`` checks it, and asks again, with the error, if it doesn't fit.
"""

import re
from typing import Annotated

import restate
from pydantic import BaseModel, Field, ValidationError
from restate import sessions
from restate.agents import openai


class Issue(BaseModel):
    """What the planner plans from: an issue's title and description."""

    title: str
    description: str


class Part(BaseModel):
    name: Annotated[str, Field(pattern=r"^[a-z0-9-]+$", description='A short name, such as "parser"')]
    title: Annotated[str, Field(description="The part, in a line: the title of its sub-issue")]
    description: Annotated[
        str, Field(description="The part, for the people following the issue: two or three plain sentences")
    ]
    files: Annotated[list[str], Field(description="The files the part owns: no other part changes them")]
    task: Annotated[str, Field(description="What to build, in full: its coder sees nothing else")]


class Plan(BaseModel):
    """A plan the parent can act on, part by part."""

    summary: Annotated[
        str, Field(description="What will be built, and how its parts fit together: a short paragraph, for people")
    ]
    parts: Annotated[list[Part], Field(min_length=1, description="Parts that can be built at the same time")]


async def ask_for[T: BaseModel](answer: type[T], ctx: sessions.SessionContext, prompt: str, *, system: str) -> T:
    """The model's answer to ``prompt``, as an ``answer``: its JSON, checked, or asked again (three times at most).

    ``openai.prompt`` answers each prompt once, in the agent's conversation: a retry of the handler gets the
    same answers, and doesn't ask the model again.
    """
    reply = await openai.prompt(ctx, prompt, system=system)
    for _ in range(3):
        try:
            return answer.model_validate_json(_json_in(reply))
        except ValidationError as error:
            reply = await openai.prompt(
                ctx, f"That doesn't fit the schema: {error}\nAnswer again, with the JSON only.", system=system
            )
    raise restate.TerminalError(f"The model gave no answer that fits {answer.__name__}")


def _json_in(reply: str) -> str:
    """The JSON object in the model's reply, without the Markdown fence it may come in."""
    found = re.search(r"\{.*\}", reply, re.DOTALL)
    return found[0] if found else reply
