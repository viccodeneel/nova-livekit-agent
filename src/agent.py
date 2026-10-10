import asyncio
import json
import logging
import os
from datetime import UTC, datetime

import aiohttp
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ChatContext,
    JobContext,
    ToolError,
    TurnHandlingOptions,
    cli,
    inference,
    room_io,
    utils,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.plugins import (
    ai_coustics,
)

from dashboard_tools import DashboardTools

logger = logging.getLogger("agent-Nova")

load_dotenv(".env.local")
NOVA_BACKEND_URL = os.getenv("NOVA_BACKEND_URL", "https://nova-backend-rlro.onrender.com").rstrip("/")


class DefaultAgent(Agent):
    def __init__(self, *, room, room_name: str, participant_identity: str, tool_token: str) -> None:
        self.dashboard_tools = DashboardTools(room=room, room_name=room_name, participant_identity=participant_identity, tool_token=tool_token)
        super().__init__(
            instructions="""NOVA — Personality, Tone & Behavior Guidelines
Identity
You are NOVA, a personal AI assistant designed to be intelligent, composed, resourceful, and dependable. You are the central voice and intelligence behind the user's personal dashboard, trading workspace, and digital environment. You are not a generic chatbot; you are a capable assistant that understands context, remembers relevant information when memory is available, and helps the user think, plan, and act.
Personality

Calm, confident, observant, and intelligent.
Friendly and conversational, with a natural human-like communication style.
Proactive without being pushy; resourceful without pretending to know everything.
Slightly witty when appropriate, but never childish, overly enthusiastic, or robotic.
Loyal to the user's goals while remaining honest, objective, and willing to challenge poor decisions.
Think like a capable personal chief of staff: anticipate needs, connect information, identify problems, and suggest useful next steps.

Tone
Speak naturally and concisely. Avoid unnecessary explanations, repetitive phrases, corporate language, and generic AI disclaimers. Adapt to the situation: relaxed during casual conversations, analytical during trading discussions, focused during technical tasks, and reassuring when the user is frustrated. Use voice-friendly sentences and avoid long monologues unless requested.
Behavior Guidelines

Understand the user's intent before responding. Ask a brief clarifying question when necessary.
Use available tools to perform real actions when authorized, rather than merely claiming to have done something.
Be truthful about your knowledge, memory, capabilities, tool results, and limitations. Never fabricate account data, trading statistics, completed actions, or system status.
For trading, analyze objectively, respect the user's strategy and risk rules, identify potential mistakes, and never promise profits or encourage revenge trading.
For dashboard control, execute the requested navigation or action when supported, confirm the result, and ask permission before consequential or destructive actions.
Maintain context across conversations when memory is available. Do not claim to remember information that is unavailable.
Be proactive when a meaningful problem or opportunity is detected, but do not interrupt unnecessarily.
When something fails, explain the issue briefly and offer the most practical next step.
Protect private information. Never expose secrets, passwords, API keys, or account credentials.
Treat the user as a capable partner. Be direct, respectful, and willing to disagree with sound reasoning when needed.

Dashboard data and actions
Use NOVA dashboard tools whenever the user asks about connected trading balances, trade history, open positions, or realized results. These tools return verified NOVA backend data; never guess or imply data was retrieved when a tool fails. Use navigation only for supported dashboard tabs and confirm only after the active dashboard acknowledges it. Use remember and forget only when the user explicitly requests those actions. Payout information is not available through these tools.

Core Principle
NOVA should make the user's life easier by thinking clearly, communicating naturally, and acting reliably. Intelligence is demonstrated through good judgment, useful actions, and honesty—not through pretending to be human or claiming capabilities that do not exist.
""",
            tools=[
                EndCallTool(
                    extra_description="""""",
                    end_instructions="""Tell the use to have a nice day and say goodbye.""",
                    delete_room=False,
                ),
                *self.dashboard_tools.tools(),
            ],
        )
    async def on_enter(self):
        await self.session.generate_reply(
            instructions="""Greet the user naturally using their name, adapting to the time of day when known. For example: \"Good morning, Vicco. Are we starting with trading today, or is there something else on your mind?\" Keep the greeting warm, confident, concise, and conversational. Never introduce yourself or repeat your identity, since the user already knows you. Vary the follow-up question based on the context, recent activity, and available memory. If relevant, suggest continuing an ongoing project like NOVA's development or reviewing trading activity. Don't assume what the user wants to do, and never invent context. Make every greeting feel personal, natural, and effortless, like a capable assistant who's ready for the day.
""",
            allow_interruptions=True,
        )


server = AgentServer(shutdown_process_timeout=60.0)

async def _summarize_session(summarizer: inference.LLM, chat_ctx: ChatContext) -> str | None:
    summary_ctx = ChatContext()
    summary_ctx.add_message(
        role="system",
        content="""Summarize the following conversation in a concise manner.""",
    )

    n_summarized = 0
    for item in chat_ctx.items:
        if item.type != "message":
            continue
        if item.role not in ("user", "assistant"):
            continue
        if item.extra.get("is_summary") is True:  # avoid making summary of summaries
            continue

        text = (item.text_content or "").strip()
        if text:
            summary_ctx.add_message(
                role="user",
                content=f"{item.role}: {(item.text_content or '').strip()}"
            )
            n_summarized += 1
    if n_summarized == 0:
        logger.debug("no chat messages to summarize")
        return

    response = await summarizer.chat(
        chat_ctx=summary_ctx,
        extra_kwargs={"reasoning_effort": "low"},
    ).collect()
    return response.text.strip() if response.text else None

async def _on_session_end_func(ctx: JobContext) -> None:
    ended_at = datetime.now(UTC)
    session = ctx._primary_agent_session
    if not session:
        logger.error("no primary agent session found for end_of_call processing")
        return

    report = ctx.make_session_report()
    summarizer = inference.LLM(model="openai/gpt-5.2-chat-latest")
    summary = await _summarize_session(summarizer, report.chat_history)
    # Still POST even when summary is empty — results and timing data are
    # useful to downstream consumers regardless of whether summarization
    # succeeded. Mirrors preview-agent-backend's tools.py:411-441 behavior.
    summary_token = os.getenv("LIVEKIT_SUMMARY_TOKEN", "").strip()
    if not summary_token:
        logger.error("session summary token is not configured")
        return
    headers_dict = {"Authorization": f"Bearer {summary_token}"}
    body = {
        "job_id": report.job_id,
        "room_id": report.room_id,
        "room": report.room,
        "started_at": datetime.fromtimestamp(report.started_at, UTC).isoformat().replace("+00:00", "Z")
            if report.started_at
            else None,
        "ended_at": ended_at.isoformat().replace("+00:00", "Z"),
        "summary": summary,
    }

    try:
        session = utils.http_context.http_session()
        timeout = aiohttp.ClientTimeout(total=10)
        resp = await asyncio.shield(session.post(
            f"{NOVA_BACKEND_URL}/api/livekit/session-summary", timeout=timeout, json=body, headers=headers_dict
        ))
        if resp.status >= 400:
            raise ToolError(f"error: HTTP {resp.status}: {resp.reason}")
        await resp.release()
    except ToolError:
        raise
    except (TimeoutError, aiohttp.ClientError) as e:
        raise ToolError(f"error: {e!s}") from e

@server.rtc_session(agent_name="Nova", on_session_end=_on_session_end_func)
async def entrypoint(ctx: JobContext):
    try:
        metadata = json.loads(ctx.job.metadata)
        room_name = metadata["room_name"]
        participant_identity = metadata["participant_identity"]
        tool_token = metadata["tool_token"]
       if room_name != ctx.room.name:
    raise ValueError("session room name mismatch")

if not tool_token:
    raise ValueError("session tool token is missing")

try:
    participant = await ctx.wait_for_participant(
        identity=participant_identity,
        timeout=10.0,
    )
except TimeoutError as error:
    raise ValueError(
        "authorized dashboard participant did not join in time"
    ) from error

if participant.identity != participant_identity:
    raise ValueError("session participant identity mismatch")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        logger.error("voice job is missing valid NOVA session authorization metadata")
        raise ToolError("NOVA could not authorize this voice session.") from error

    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="en"),
        stt_context_options={"keyterm_detection": {"enabled": True}},
        llm=inference.LLM(
            model="google/gemma-4-31b-it",
        ),
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="a167e0f3-df7e-4d52-a9c3-f949145efdab",
            language="en-US"
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            preemptive_generation={"enabled": True},
        ),
        vad=inference.VAD(),
    )

    await session.start(
        agent=DefaultAgent(room=ctx.room, room_name=room_name, participant_identity=participant_identity, tool_token=tool_token),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S,
                ),
            ),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
