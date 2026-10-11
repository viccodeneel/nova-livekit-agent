import json
import logging
import os
import time
import uuid
from typing import Any, Literal

import aiohttp
from livekit.agents import RunContext, ToolError, function_tool, utils

logger = logging.getLogger("agent-Nova.dashboard-tools")

PAGES = {"overview", "trade-journal", "analytics", "accounts", "ai", "finance", "settings"}

def parse_navigation_ack(response: str, expected_page: str) -> dict[str, Any]:
    """Accept only an explicit success acknowledgement from the dashboard browser."""
    result = json.loads(response)
    if expected_page not in PAGES or result.get("success") is not True or result.get("page") != expected_page:
        raise ValueError("dashboard navigation was not confirmed")
    return result


def latest_user_message(history: Any) -> str:
    """Return the most recent user utterance from the actual agent session history."""
    for item in reversed(history.items):
        if item.type == "message" and item.role == "user":
            return (item.text_content or "").strip()[:2000]
    return ""


class DashboardTools:
    """NOVA voice tools using only the scoped backend capability for this room."""

    def __init__(self, *, room: Any, room_name: str, participant_identity: str, tool_token: str) -> None:
        self.room = room
        self.room_name = room_name
        self.participant_identity = participant_identity
        self.tool_token = tool_token
        self.backend_url = os.getenv("NOVA_BACKEND_URL", "https://nova-backend-rlro.onrender.com").rstrip("/")
        self.bridge_secret = os.getenv("LIVEKIT_AGENT_BRIDGE_SECRET", "").strip()

    def tools(self) -> list[Any]:
        return [
            self.get_account_balance,
            self.get_trade_history,
            self.get_open_positions,
            self.get_realized_results,
            self.remember,
            self.recall_memories,
            self.update_memory,
            self.forget,
            self.navigate_to_tab,
        ]

    async def _refresh_tool_token(self, session: aiohttp.ClientSession) -> None:
        if len(self.bridge_secret) < 32:
            raise ToolError("NOVA voice tools are not configured on the agent service.")
        url = f"{self.backend_url}/api/livekit/voice/agent/credentials"
        try:
            async with session.post(
                url,
                json={"roomName": self.room_name, "agentIdentity": self.room.local_participant.identity},
                headers={"Authorization": f"Bearer {self.bridge_secret}"},
                timeout=aiohttp.ClientTimeout(total=8),
            ) as response:
                payload = await response.json(content_type=None)
                if response.status != 200 or not payload.get("success"):
                    raise ToolError("NOVA could not authorize this voice session for dashboard tools.")
                data = payload.get("data") or {}
                if data.get("roomName") != self.room_name or data.get("participantIdentity") != self.participant_identity:
                    raise ToolError("NOVA returned credentials for a different voice session.")
                self.tool_token = str(data.get("toolToken") or "")
                if not self.tool_token:
                    raise ToolError("NOVA did not return a voice tool credential.")
        except ToolError:
            raise
        except (TimeoutError, aiohttp.ClientError, ValueError) as error:
            raise ToolError("NOVA could not authorize dashboard tools. Try again shortly.") from error

    def _token_expires_soon(self) -> bool:
        try:
            payload = self.tool_token.split(".")[1]
            claims = json.loads(__import__("base64").urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
            return int(claims.get("exp", 0)) < int(time.time()) + 30
        except (IndexError, ValueError, TypeError, AttributeError):
            return True

    async def _call(self, name: str, arguments: dict[str, Any], user_message: str) -> str:
        if name == "recall_memories":
            logger.info("NOVA voice memory recall tool invoked.")
        if self._token_expires_soon():
            session = utils.http_context.http_session()
            await self._refresh_tool_token(session)
        request_id = str(uuid.uuid4())
        body = {
            "name": name,
            "arguments": arguments,
            "user_message": user_message[:2000],
            "request_id": request_id,
        }
        url = f"{self.backend_url}/api/livekit/voice/agent/tools"
        session = utils.http_context.http_session()
        try:
            async with session.post(
                url,
                json=body,
                headers={"Authorization": f"Bearer {self.tool_token}"},
                timeout=aiohttp.ClientTimeout(total=12),
            ) as response:
                payload = await response.json(content_type=None)
                if response.status == 401:
                    await self._refresh_tool_token(session)
                    async with session.post(
                        url,
                        json=body,
                        headers={"Authorization": f"Bearer {self.tool_token}"},
                        timeout=aiohttp.ClientTimeout(total=12),
                    ) as retry:
                        payload = await retry.json(content_type=None)
                        response_status = retry.status
                else:
                    response_status = response.status
            if response_status != 200 or not payload.get("success"):
                message = payload.get("error") if isinstance(payload, dict) else None
                logger.warning("NOVA voice tool rejected (%s, HTTP %s).", name, response_status)
                raise ToolError(message if isinstance(message, str) else "NOVA could not retrieve that dashboard data.")
            if name == "recall_memories":
                data = payload.get("data") if isinstance(payload, dict) else None
                memories = data.get("memories") if isinstance(data, dict) else None
                logger.info("NOVA voice memory recall completed (%s matches).", len(memories) if isinstance(memories, list) else 0)
            return json.dumps(payload.get("data"), ensure_ascii=False, separators=(",", ":"))
        except ToolError:
            raise
        except (TimeoutError, aiohttp.ClientError, ValueError) as error:
            logger.warning("NOVA voice tool request failed (%s).", name)
            raise ToolError("NOVA's dashboard data service is unavailable. Please try again shortly.") from error

    @function_tool()
    async def get_account_balance(self, context: RunContext) -> str:
        """Get real connected trading-account balances, equity, status, and data freshness from NOVA."""
        return await self._call("get_account_info", {}, latest_user_message(context.session.history))

    @function_tool()
    async def get_trade_history(
        self,
        context: RunContext,
        period: str | None = None,
        symbol: str | None = None,
        limit: int = 10,
    ) -> str:
        """Retrieve real recorded trading history. Optional period: today, yesterday, this_week, last_week, this_month, last_month, or all. Limit is 1-25."""
        arguments: dict[str, Any] = {"limit": limit}
        if period is not None:
            arguments["period"] = period
        if symbol is not None:
            arguments["symbol"] = symbol
        return await self._call("get_trades", arguments, latest_user_message(context.session.history))

    @function_tool()
    async def get_open_positions(self, context: RunContext) -> str:
        """Retrieve currently open positions from NOVA's connected trading accounts."""
        return await self._call("get_open_positions", {}, latest_user_message(context.session.history))

    @function_tool()
    async def get_realized_results(
        self,
        context: RunContext,
        period: str = "today",
        date: str | None = None,
    ) -> str:
        """Retrieve real realized trading results for a supported period or UTC date."""
        arguments: dict[str, Any] = {"period": period}
        if date is not None:
            arguments["date"] = date
        return await self._call("get_pnl", arguments, latest_user_message(context.session.history))

    @function_tool()
    async def remember(
        self,
        context: RunContext,
        content: str,
        kind: Literal["preference", "goal", "fact", "rule", "project", "event"],
        evidence: str,
    ) -> str:
        """Save a short memory only when the user explicitly asked NOVA to remember it. Include the exact words from that request as evidence."""
        return await self._call("remember", {"content": content, "kind": kind, "evidence": evidence}, latest_user_message(context.session.history))

    @function_tool()
    async def recall_memories(self, context: RunContext, query: str) -> str:
        """Retrieve relevant saved NOVA notes for the current topic. Returns only relevant notes, if any."""
        return await self._call("recall_memories", {"query": query}, latest_user_message(context.session.history))

    @function_tool()
    async def update_memory(
        self,
        context: RunContext,
        description: str,
        content: str,
        evidence: str,
        kind: Literal["preference", "goal", "fact", "rule", "project", "event"] | None = None,
    ) -> str:
        """Update one saved NOVA note only when the user explicitly asks you to update or correct what you remember. Include exact words as evidence."""
        arguments: dict[str, Any] = {"description": description, "content": content, "evidence": evidence}
        if kind is not None:
            arguments["kind"] = kind
        return await self._call("update_memory", arguments, latest_user_message(context.session.history))

    @function_tool()
    async def forget(self, context: RunContext, description: str) -> str:
        """Forget matching NOVA memories only when the user explicitly asks to forget or delete them."""
        return await self._call("forget", {"description": description}, latest_user_message(context.session.history))

    @function_tool()
    async def navigate_to_tab(
        self,
        context: RunContext,
        page: Literal["overview", "trade-journal", "analytics", "accounts", "ai", "finance", "settings"],
    ) -> str:
        """Navigate the active authenticated NOVA dashboard to an existing tab. Confirm only after the browser acknowledges the navigation."""
        if page not in PAGES:
            raise ToolError("That NOVA dashboard tab is not supported.")
        try:
            response = await self.room.local_participant.perform_rpc(
                destination_identity=self.participant_identity,
                method="nova.navigate",
                payload=json.dumps({"page": page}),
                response_timeout=10,
            )
            parse_navigation_ack(response, page)
            return json.dumps({"success": True, "page": page})
        except ToolError:
            raise
        except Exception as error:
            raise ToolError("The NOVA dashboard could not confirm navigation. Please try again.") from error
