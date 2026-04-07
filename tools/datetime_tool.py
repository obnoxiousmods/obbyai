"""Date/time and timezone tool."""
import asyncio
from datetime import datetime, timezone
import time

TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "get_datetime",
        "description": (
            "Get the current date and time. Use when the user asks what time or date it is, "
            "for scheduling, deadlines, or any time-sensitive context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Timezone name (e.g. 'UTC', 'US/Pacific', 'Europe/London'). Defaults to server local time.",
                    "default": "local",
                }
            },
            "required": [],
        },
    },
}


async def run(timezone_name: str = "local") -> dict:
    now_utc = datetime.now(tz=timezone.utc)
    now_local = datetime.now()
    result = {
        "utc": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "local": now_local.strftime("%Y-%m-%d %H:%M:%S"),
        "unix_timestamp": int(time.time()),
        "day_of_week": now_local.strftime("%A"),
        "iso8601": now_utc.isoformat(),
    }
    if timezone_name and timezone_name != "local":
        try:
            import zoneinfo
            tz = zoneinfo.ZoneInfo(timezone_name)
            now_tz = datetime.now(tz=tz)
            result["requested_timezone"] = {
                "timezone": timezone_name,
                "datetime": now_tz.strftime("%Y-%m-%d %H:%M:%S %Z"),
            }
        except Exception:
            result["timezone_error"] = f"Unknown timezone: {timezone_name}"
    return result
