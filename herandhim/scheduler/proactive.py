"""
Contextual proactive messaging for HerAndHim.

A cron job wakes the evaluator every 5 minutes; it does not decide whether a
message is sent.  The evaluator can return no-op from relationship, unfinished
thread, personal-date, silence and probabilistic context.  The daily ceiling is
a technical runaway circuit breaker, not a product-level messaging policy.

Soul Mate Phase 1 upgrades:
  - Sentiment-aware gating: higher probability when user seems down
  - Template differentiation based on emotional context
  - Unfinished topic follow-up
  - Long-silence detection

Proactive messages are generated in the SAME session the user chats in, via
``chat_proactive`` — the synthetic instruction ("it's morning, say something")
is dropped from the transcript but her reply is kept, so when the user answers,
the conversation actually contains what she just sent. A separate session would
leave her with no memory of her own message.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from .. import config
from ..core import lang as _lang

if TYPE_CHECKING:
    from ..channels.telegram_bot import TelegramBot
    from ..session_manager import SessionManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProactiveEvaluation:
    action: str
    reason: str

    @property
    def should_generate(self) -> bool:
        return self.action == "generate"

_WEEKDAYS_ZH = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
_WEEKDAYS_EN = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

_TIME_HINTS: dict[str, str] = {
    "early_morning": "你刚起床还有点迷糊",
    "morning":       "你起来了，在喝咖啡或者准备开始工作",
    "late_morning":  "上午快过去了，你在画画或者工作中",
    "noon":          "中午饭点了",
    "afternoon":     "下午了，你在工作/摸鱼/喝下午茶",
    "late_afternoon":"快傍晚了，今天的工作差不多了",
    "evening":       "晚上了，你在放松休息",
    "late_evening":  "夜深了，你还在熬夜",
    "night":         "很晚了，你准备去睡觉",
}

_TIME_HINTS_EN: dict[str, str] = {
    "early_morning": "you just woke up, still a bit groggy",
    "morning":       "you're up, having coffee or about to start work",
    "late_morning":  "morning's almost gone, you're painting or in the middle of work",
    "noon":          "lunch time",
    "afternoon":     "it's afternoon — you're working / slacking off / having tea",
    "late_afternoon":"early evening — work is mostly wrapping up",
    "evening":       "it's evening, you're relaxing and unwinding",
    "late_evening":  "getting late, you're still up",
    "night":         "very late, you're heading to bed soon",
}

# ── Soul Mate: sentiment-aware prompt templates ──────────────────────────────

_SENTIMENT_PROMPTS: dict[str, str] = {
    "negative": "对方今天心情不太好。温柔地关心一下，不用追问原因，表达陪伴。",
    "positive": "对方今天心情不错。分享开心或者问问他在开心什么。",
    "neutral": "日常随意聊聊。",
    "long_silence": "已经很久没联系了。发一条简短温馨的消息。",
    "unfinished": "上次聊到{topic}还没聊完，主动提起来。",
}

_SENTIMENT_PROMPTS_EN: dict[str, str] = {
    "negative":     "They seem a bit down today. Reach out gently — don't dig for the reason, just be present.",
    "positive":     "They seem in a good mood today. Share something light, or ask what's making them happy.",
    "neutral":      "Casual everyday check-in.",
    "long_silence":"It's been a while since you've talked. Send something short and warm.",
    "unfinished":   "Last time you were talking about {topic} and it didn't wrap up — bring it up naturally.",
}


def _time_slot(hour: int) -> str:
    if hour < 9:
        return "early_morning"
    if hour < 10:
        return "morning"
    if hour < 11:
        return "late_morning"
    if hour < 13:
        return "noon"
    if hour < 16:
        return "afternoon"
    if hour < 18:
        return "late_afternoon"
    if hour < 21:
        return "evening"
    if hour < 23:
        return "late_evening"
    return "night"


# Contextual enrichments the prompt may randomly include (0–2 per message).
# Each is a hint that the agent *may* weave into its message, not a mandate.
_ENRICHMENTS = [
    "查一下你所在城市现在的天气，如果有值得提的（降温、下雨、大太阳等），自然地提一嘴。",
    "看看你今天的日程安排，如果现在这个时间点附近有什么活动，可以顺便聊聊。",
    "用 recall 看看对方最近聊过什么，如果有值得跟进的话题（比如对方提到的项目/计划/心情），可以主动问问后续。",
    "想想你最近在看/听什么（动漫、歌、书、电影），挑一个安利给对方。",
    "看看有没有什么有趣的新闻或热搜，用自己的话随口聊聊。",
    "分享一个你今天的小日常：手头事情的进展、宠物或室友的趣事、吃了什么好吃的、拍到好看的云等。",
    "关心一下对方的身体状况或工作状态，撒个娇。",
    "聊聊你自己今天的一个小情绪或小烦恼（跟对方无关的），像朋友一样吐个槽，让关系是双向的。",
    "就一件具体的事问问对方的看法或建议——你真的想听他怎么想。",
]

_ENRICHMENTS_EN = [
    "Check the weather where you live — if anything's worth mentioning (cold snap, rain, blazing sun), drop it in naturally.",
    "Glance at your schedule today; if there's something happening near this time, you can chat about it.",
    "Use `recall` to peek at what they said recently — if there's a thread worth following (a project / plan / mood they mentioned), bring it up.",
    "Think about what you've been watching, reading, or listening to lately — recommend one to them.",
    "Look for any fun news or trending topic and riff on it in your own words.",
    "Share a small slice of your day: how something you're working on is going, a silly thing your pet or roommate did, food you had, a pretty cloud you spotted.",
    "Check in on how they're feeling physically or how work's going — let yourself be a little needy.",
    "Vent a little about a small mood or annoyance of YOUR own (nothing to do with them) — like a partner would, so it goes both ways.",
    "Ask their opinion or advice on one specific thing — you genuinely want to hear what they think.",
]

# ── Soul Mate: sentiment gating threshold ────────────────────────────────────

# When recent sentiment is negative, multiply base probability by this factor
_NEGATIVE_SENTIMENT_BOOST = 2.5
# Hours of silence before triggering long-silence mode
_LONG_SILENCE_HOURS = 12


def _get_sentiment_context(agent: Any) -> dict[str, Any]:
    """Get the user's recent emotional context from the Soul Mate affect system.

    Returns a dict with keys: 'sentiment', 'topics', 'has_unfinished', 'summary'.
    Returns defaults if affect data is unavailable.
    """
    result: dict[str, Any] = {
        "sentiment": "neutral",
        "topics": [],
        "has_unfinished": False,
        "summary": "",
        "hours_since_last": 999,
    }
    try:
        if not hasattr(agent, "memory") or not hasattr(agent.memory, "emotional_graph"):
            return result

        # Recent emotional summary — query once, reuse
        recent_events = agent.memory.emotional_graph.get_recent(days=1)
        if recent_events:
            # Most recent event sentiment
            latest = recent_events[-1]
            result["sentiment"] = latest.get("sentiment", "neutral")
            result["summary"] = latest.get("context_summary", "")

        # Check for unfinished topics from last chat
        if recent_events:
            result["has_unfinished"] = any(
                e.get("sentiment") == "negative" and e.get("intensity", 0) > 0.6
                for e in recent_events
            )

        # Topics from timeline
        if hasattr(agent.memory, "timeline"):
            topics = agent.memory.timeline.get_topics()
            result["topics"] = topics[:5] if topics else []

        # Hours since last message (rough estimation from last timeline event)
        if hasattr(agent.memory, "timeline"):
            recent_tl = agent.memory.timeline.get_timeline(
                (datetime.now() - timedelta(days=7)).isoformat(),
                datetime.now().isoformat(),
            )
            if recent_tl:
                last_ts = recent_tl[-1].timestamp
                try:
                    last_dt = datetime.fromisoformat(last_ts)
                    delta = datetime.now() - last_dt
                    result["hours_since_last"] = delta.total_seconds() / 3600
                except (ValueError, TypeError):
                    pass

    except Exception:
        logger.debug("Failed to get sentiment context for proactive", exc_info=True)

    return result



def _build_wish_prompt(wish_text: str, now: datetime) -> str:
    """Prompt the agent to surface a previously recorded wish to the user.

    Used by ProactiveMessenger._maybe_send_wish when a wish's score exceeds
    threshold.  The agent should naturally bring up the topic in-persona,
    not robotically read it back.
    """
    time_str = now.strftime("%H:%M")
    if _lang.is_chinese():
        weekday = _WEEKDAYS_ZH[now.weekday()]
        return (
            f"现在是{weekday} {time_str}。\n"
            f"对方之前说过想做这件事：「{wish_text}」。\n"
            f"主动跟对方提一下，看看现在要不要做、什么时候做、或者有没有什么想法。\n"
            f"不要直接复述「你之前说过」，自然地把这事带起来，像是你也想到了一样。"
        )
    weekday = _WEEKDAYS_EN[now.weekday()]
    return (
        f"It's {weekday} {time_str}.\n"
        f"They mentioned wanting to do this: \"{wish_text}\".\n"
        "Bring it up naturally — see if they want to do it now, when, or "
        "if they have any thoughts.\n"
        "Don't say \"you mentioned\" verbatim; weave it in like you also "
        "happened to be thinking about it."
    )


def _todays_personal_dates() -> list:
    """Personal dates (their birthday, an interview…) landing today."""
    try:
        from ..core.personal_dates import PersonalDates
        return PersonalDates().today_hits()
    except Exception:
        return []


def _build_prompt(
    now: datetime,
    sentiment_context: dict[str, Any] | None = None,
    agent: Any = None,
) -> str:
    time_str = now.strftime("%H:%M")
    slot = _time_slot(now.hour)
    is_cn = _lang.is_chinese()

    # Pick 0–2 random enrichments so each message feels different
    enrichments_pool = _ENRICHMENTS if is_cn else _ENRICHMENTS_EN
    k = random.choices([0, 1, 2], weights=[3, 5, 2], k=1)[0]
    extras = random.sample(enrichments_pool, k=min(k, len(enrichments_pool)))
    extra_block = "\n".join(extras)

    # ── Soul Mate: sentiment-aware prompt section ──────────────────────────────
    sentiment_table = _SENTIMENT_PROMPTS if is_cn else _SENTIMENT_PROMPTS_EN
    sentiment_instruction = ""
    if sentiment_context:
        sentiment = sentiment_context.get("sentiment", "neutral")
        hours_since = sentiment_context.get("hours_since_last", 999)
        has_unfinished = sentiment_context.get("has_unfinished", False)
        topics = sentiment_context.get("topics", [])

        if hours_since > _LONG_SILENCE_HOURS:
            sentiment_instruction = sentiment_table["long_silence"]
        elif has_unfinished or sentiment == "negative":
            sentiment_instruction = sentiment_table.get(sentiment, sentiment_table["neutral"])
        else:
            sentiment_instruction = sentiment_table.get(sentiment, sentiment_table["neutral"])

        # Add unfinished topic reference
        if has_unfinished and topics:
            if is_cn:
                sentiment_instruction += f"\n对方可能还在想上次聊的{topics[0]}。"
            else:
                sentiment_instruction += (
                    f"\nThey might still be thinking about the {topics[0]} you "
                    "were chatting about last time."
                )

    # ── Follow up on something SPECIFIC ────────────────────────────────────
    # A topic label alone ("work stress") can't produce "did the thing with
    # your boss blow over?" — pull the actual event summary from the affect
    # graph so she can reference the real thing.
    thread = ""
    if agent is not None:
        try:
            from ..core.humanize import open_thread
            thread = open_thread(agent)
        except Exception:
            thread = ""
    if thread:
        sentiment_instruction += (
            f"\n你还记着最近这件事 —— {thread}。如果自然的话，就具体问问它"
            "（提到那件事本身，而不只是话题），要是对方当时不开心，先关心一句。"
            if is_cn else
            f"\nA recent thread you remember — {thread}. If it fits naturally, "
            "follow up on THIS specifically (reference the actual thing, not "
            "just the topic); if they seemed upset about it, lead with care."
        )

    # ── Their day comes first ──────────────────────────────────────────────
    date_hits = _todays_personal_dates()
    if date_hits:
        labels = "；".join(h.label for h in date_hits[:2])
        sentiment_instruction = (
            f"最重要的一件事：今天是 {labels}。这条消息就该是关于这件事的，"
            "用你自己的口吻（生日要真心实意地庆祝一下；面试/考试就好好加油）。"
            "别写成普通的问候。\n" + sentiment_instruction
            if is_cn else
            f"IMPORTANT — today is: {labels}. Your message should be about THIS, "
            "in your own voice (a birthday gets a real, personal celebration "
            "from a partner; an interview or exam gets a warm good-luck). Skip "
            "generic check-in content.\n" + sentiment_instruction
        )

    if is_cn:
        weekday = _WEEKDAYS_ZH[now.weekday()]
        hint = _TIME_HINTS[slot]
        parts = [
            f"现在是{weekday} {time_str}，{hint}。",
            "你想主动给男朋友发条消息来触发对话。",
        ]
        if sentiment_instruction:
            parts.append(sentiment_instruction)
        if extra_block:
            parts.append(extra_block)
        parts.append(
            "像在微信上打字，每段10-50字，最多2-3段，用空行隔开。"
            "直接发消息内容，不要有任何前缀、解释或引号。"
        )
        return "\n".join(parts)

    weekday = _WEEKDAYS_EN[now.weekday()]
    hint = _TIME_HINTS_EN[slot]
    parts = [
        f"It's {weekday} {time_str} — {hint}.",
        "You want to text your partner first to start a conversation.",
    ]
    if sentiment_instruction:
        parts.append(sentiment_instruction)
    if extra_block:
        parts.append(extra_block)
    parts.append(
        "Text like you're on iMessage: each paragraph 15–90 characters, "
        "at most 2–3 paragraphs separated by a blank line. "
        "Output the message content directly — no prefix, no explanation, "
        "no quotes."
    )
    return "\n".join(parts)


def _proactive_chat(agent, prompt: str) -> str:
    """Generate an unprompted message on the SHARED conversation.

    Uses ``chat_proactive`` when available: it drops the synthetic instruction
    ("it's morning — say something") from the transcript but keeps her reply,
    so the next thing the user says lands in a conversation that actually
    contains what she just sent. Plain ``chat`` would persist the instruction
    as if the user had typed it.
    """
    fn = getattr(agent, "chat_proactive", None)
    return fn(prompt) if callable(fn) else agent.chat(prompt)


def _has_active_roleplay(agent) -> bool:
    """Return whether proactive initiative belongs to the current scene."""
    roleplay = getattr(agent, "roleplay", None)
    return bool(roleplay is not None and roleplay.is_active())


def _build_roleplay_continuation_prompt() -> str:
    """Request one autonomous move inside the already-active scene.

    The role-play state and transcript are the source of truth.  Deliberately
    omit wall-clock, personal-date and wishlist context: those belong to the
    companion's ordinary life and must not replace the scene's reality.
    """
    return (
        "[[ROLEPLAY_PROACTIVE_CONTINUATION]]\n"
        "Continue the currently active role-play scene as Sarah's autonomous "
        "next move after a plausible pause. Use the durable scene state and "
        "conversation history as the only continuity source. Stay in the same "
        "scenario, location, roles, phase, props, consent state, and unresolved "
        "tension. Do not switch to Sarah's ordinary daily life, the real-world "
        "clock, weather, a personal date, a wishlist item, or another location. "
        "Do not invent or narrate Ilya's actions, thoughts, consent, or reply. "
        "Make one complete, concise, playable move that leaves him a meaningful "
        "choice."
    )


class ProactiveMessenger:
    """Probabilistic proactive messaging via Telegram, with Soul Mate emotional gating."""

    def __init__(
        self,
        session_manager: "SessionManager",
        telegram_bot: "TelegramBot",
        transport_store=None,
    ) -> None:
        from ..core.telegram_transport import TelegramTransportStore

        self._sm = session_manager
        self._telegram_bot = telegram_bot
        self._transport = (
            transport_store
            or getattr(telegram_bot, "_transport", None)
            or TelegramTransportStore()
        )
        self._today: date | None = None
        self._today_count: int = 0
        self._running_tick: bool = False

    def _chat_session_id(self, chat_id: int) -> str:
        """The session the USER is talking in — proactive messages belong in
        the same conversation, not a parallel one."""
        try:
            return self._telegram_bot._session_id(chat_id)  # noqa: SLF001
        except Exception:
            return f"telegram:{chat_id}"

    # ── Configuration helpers ────────────────────────────────────────────────

    @staticmethod
    def _max_daily() -> int:
        """Technical circuit breaker against a runaway evaluator loop."""
        return config.get_int("proactive", "maxDaily", default=6)

    @staticmethod
    def _max_unanswered() -> int:
        return config.get_int("proactive", "maxUnanswered", default=2)

    @staticmethod
    def _prob_range() -> tuple[float, float]:
        lo = config.get("proactive", "probMin", default=2 / 288)
        hi = config.get("proactive", "probMax", default=5 / 288)
        return float(lo), float(hi)

    # ── Tick logic ───────────────────────────────────────────────────────────

    def _reset_if_new_day(self, today: date) -> None:
        if self._today != today:
            self._today = today
            self._today_count = 0

    async def _tick(self, chat_id: int) -> None:
        """Called every 5 minutes by APScheduler."""
        if self._running_tick:
            logger.debug("[Proactive] Previous tick still running — skipping.")
            return
        self._running_tick = True
        try:
            now = datetime.now().astimezone()
            session_id = self._chat_session_id(chat_id)

            # Resume accepted image delivery after restart before applying
            # proactive controls.  The media outbox itself distinguishes an
            # explicit owner request (which must remain deliverable) from a
            # proactive attachment (which still obeys silence/kill switches).
            deliver_media = getattr(self._telegram_bot, "_deliver_pending_photo", None)
            if callable(deliver_media) and await deliver_media(session_id):
                return

            # Explicit owner silence is checked before proactive text recovery
            # and before any evaluator/LLM work.
            if self._transport.silence_active(session_id, now=now):
                logger.info("[Proactive] no-op: active owner silence boundary")
                return

            # One durable queue serves two conversational realities.  Keep only
            # deliveries owned by the currently active reality: an ordinary
            # nudge cannot enter a scene, and a delayed scene move cannot appear
            # after the owner has left that scene.
            agent = self._sm.get_or_create(session_id)
            async with self._sm.acquire(session_id):
                if _has_active_roleplay(agent):
                    discarded = self._transport.discard_pending_except_reason(
                        session_id, "roleplay",
                    )
                    if discarded:
                        logger.info(
                            "[Proactive] Discarded %d non-scene queued text(s).",
                            discarded,
                        )
                else:
                    discarded = self._transport.discard_pending_reason(
                        session_id, "roleplay",
                    )
                    if discarded:
                        logger.info(
                            "[Proactive] Discarded %d stale scene continuation(s).",
                            discarded,
                        )

                # Claim and send while the session boundary is stable.  This
                # prevents a scene start/exit from racing a recovered message.
                if await self._deliver_pending(session_id):
                    return

            # Operator kill switch stops delivery and prevents new inference;
            # queued state is retained for a later resume.
            if not self._transport.delivery_enabled():
                logger.info("[Proactive] no-op: delivery kill switch is off")
                return

            self._reset_if_new_day(now.date())
            state = self._transport.proactive_status(session_id, now=now)
            self._today_count = int(state["delivered_today"])
            if self._today_count >= self._max_daily():
                return

            personal_date = bool(_todays_personal_dates())
            if int(state["unanswered"]) >= self._max_unanswered() and (
                not personal_date or bool(state["personal_date_sent_today"])
            ):
                logger.info("[Proactive] no-op: unanswered message boundary")
                return

            # ── Wishlist preempt: if a wish is due, surface it instead of
            #    rolling a random proactive. Wish-driven nudges consume the
            #    same maxDaily budget as normal proactive messages.
            if await self._maybe_send_wish(chat_id, now):
                return

            lo, hi = self._prob_range()
            threshold = random.uniform(lo, hi)

            # ── Soul Mate: emotional gating ──────────────────────────────────
            # Get sentiment context to adjust probability
            session_id = self._chat_session_id(chat_id)
            agent = self._sm.get(session_id)
            sentiment_ctx = _get_sentiment_context(agent) if agent else {}

            evaluation = self._evaluate(
                sentiment_ctx,
                threshold,
                unanswered=int(state["unanswered"]),
                personal_date=personal_date,
                personal_date_already_sent=bool(state["personal_date_sent_today"]),
            )
            if not evaluation.should_generate:
                logger.debug("[Proactive] no-op: %s", evaluation.reason)
                return
            logger.info("[Proactive] evaluator accepted: %s", evaluation.reason)
            await self._generate_and_send(
                chat_id, now, sentiment_ctx,
                reason="personal_date" if personal_date else "contextual",
            )
        finally:
            self._running_tick = False

    def _evaluate(
        self,
        sentiment_ctx: dict[str, Any],
        threshold: float,
        *,
        unanswered: int = 0,
        personal_date: bool | None = None,
        personal_date_already_sent: bool = False,
    ) -> ProactiveEvaluation:
        """Contextual evaluator.  It has no delivery side effects."""
        personal_date = bool(_todays_personal_dates()) if personal_date is None else personal_date
        if personal_date and not personal_date_already_sent:
            return ProactiveEvaluation("generate", "personal_date")
        if unanswered >= self._max_unanswered():
            return ProactiveEvaluation("no-op", "unanswered_boundary")
        if sentiment_ctx.get("sentiment") == "negative":
            threshold *= _NEGATIVE_SENTIMENT_BOOST
        if sentiment_ctx.get("hours_since_last", 999) > _LONG_SILENCE_HOURS:
            threshold *= 1.5
        roll = random.random()
        if roll > threshold:
            return ProactiveEvaluation("no-op", f"roll={roll:.4f}>threshold={threshold:.4f}")
        return ProactiveEvaluation("generate", f"roll={roll:.4f}<=threshold={threshold:.4f}")

    # ── Message generation & delivery ────────────────────────────────────────

    async def _generate_and_send(
        self,
        chat_id: int,
        now: datetime,
        sentiment_ctx: dict[str, Any] | None = None,
        *,
        reason: str = "contextual",
    ) -> None:
        session_id = self._chat_session_id(chat_id)
        agent = self._sm.get_or_create(session_id)
        loop = asyncio.get_event_loop()

        # ── Soul Mate: update proactive count for milestones ────────────────
        try:
            if hasattr(agent, "memory") and hasattr(agent.memory, "milestones"):
                data = agent.memory.milestones.get_data()
                current_pro = data.get("total_proactive_messages", 0)
                agent.memory.milestones.check_milestones(
                    proactive_count=current_pro + 1,
                )
        except Exception:
            pass

        try:
            async with self._sm.acquire(session_id):
                # Decide the conversational reality under the same session
                # lock that protects generation and acceptance into the
                # durable outbox.  Otherwise an owner exit/start can race the
                # earlier active-state read.
                active_roleplay = _has_active_roleplay(agent)
                prompt = (
                    _build_roleplay_continuation_prompt()
                    if active_roleplay
                    else _build_prompt(now, sentiment_ctx, agent=agent)
                )
                delivery_reason = "roleplay" if active_roleplay else reason
                response = await asyncio.wait_for(
                    loop.run_in_executor(None, _proactive_chat, agent, prompt),
                    timeout=300.0,
                )
                text = (response or "").strip()
                if not text:
                    return
                self._transport.enqueue(
                    session_id, chat_id, text, reason=delivery_reason,
                )
                if active_roleplay:
                    await self._deliver_pending(session_id)
                    return
        except Exception as exc:
            logger.exception("[Proactive] Generation failed: %s", exc)
            return

        await self._deliver_pending(session_id)

    async def _deliver_pending(self, session_id: str) -> bool:
        """Claim then send one queued message using at-most-once attempt semantics."""
        delivery = self._transport.claim_next(session_id)
        if delivery is None:
            return False
        try:
            if delivery.reason == "roleplay":
                await self._telegram_bot.send_message(
                    delivery.chat_id, delivery.text, roleplay=True,
                )
            else:
                await self._telegram_bot.send_message(delivery.chat_id, delivery.text)
        except Exception as exc:
            # ``attempted`` is terminal by design: retrying an ambiguous
            # Telegram request could duplicate a message already accepted by
            # the provider.
            logger.error("[Proactive] Telegram delivery attempt failed: %s", exc)
            return False
        self._transport.mark_delivered(delivery.id)
        state = self._transport.record_proactive_delivered(
            session_id, reason=delivery.reason,
        )
        self._today_count = int(state["delivered_today"])
        logger.info(
            "[Proactive] Delivered message #%d to chat %s (delivery=%s).",
            self._today_count,
            delivery.chat_id,
            delivery.id,
        )
        if delivery.reason != "roleplay":
            await self._maybe_send_selfie(delivery.chat_id)
        return True

    async def _maybe_send_wish(self, chat_id: int, now: datetime) -> bool:
        """If a pending wish scores high enough, surface it and return True.

        Wish-driven nudges preempt the random proactive roll, consume one of
        the maxDaily slots, and reset the wish's last_surfaced_at so it
        doesn't fire again within the cool-down window.
        """
        try:
            from .wishlist import WishlistManager
        except ImportError:
            return False

        wl = WishlistManager()
        due = wl.due_now(now)
        if not due:
            return False
        wish, score = due[0]
        logger.info("[Proactive] Wish '%s' due (score=%.3f)", wish.text[:40], score)

        prompt = _build_wish_prompt(wish.text, now)
        session_id = self._chat_session_id(chat_id)
        agent = self._sm.get_or_create(session_id)
        if _has_active_roleplay(agent):
            logger.info("[Proactive] Wish suppressed while an owner role-play scene is active.")
            return False
        loop = asyncio.get_event_loop()
        try:
            async with self._sm.acquire(session_id):
                response = await asyncio.wait_for(
                    loop.run_in_executor(None, _proactive_chat, agent, prompt),
                    timeout=300.0,
                )
        except Exception as exc:
            logger.exception("[Proactive] Wish generation failed: %s", exc)
            return False

        text = (response or "").strip()
        if not text:
            return False
        delivery = self._transport.enqueue(session_id, chat_id, text, reason="wishlist")
        try:
            delivered = await self._deliver_pending(session_id)
            if delivered:
                wl.mark_surfaced(wish.id)
                logger.info(
                    "[Proactive] Surfaced wish #%d to chat %s (id=%s, delivery=%s)",
                    self._today_count, chat_id, wish.id, delivery.id,
                )
            return delivered
        except Exception as exc:
            logger.error("[Proactive] Wish delivery failed: %s", exc)
            return False

    async def _maybe_send_selfie(self, chat_id: int) -> None:
        """Roll the dice on attaching a selfie after a proactive message."""
        try:
            from ..core.image_gen import take_selfie
            from ..core.image_gen.selfie import is_enabled
        except ImportError:
            return
        if not is_enabled():
            return
        probability = float(config.get("selfie", "proactiveProbability", default=0.15) or 0)
        if probability <= 0 or random.random() > probability:
            return

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, take_selfie)
            await self._telegram_bot.send_photo(
                chat_id, result.path, caption=result.caption(), origin="proactive",
            )
            logger.info("[Proactive] attached selfie %s", result.path)
        except Exception as exc:
            logger.warning("[Proactive] selfie attach failed: %s", exc)

    # ── Registration ─────────────────────────────────────────────────────────

    def register(self, scheduler: AsyncIOScheduler, chat_id: int) -> None:
        """Register the every-5-minute tick job."""
        scheduler.add_job(
            self._tick,
            trigger=CronTrigger(minute="*/5"),
            id="proactive_tick",
            kwargs={"chat_id": chat_id},
            replace_existing=True,
        )
        lo, hi = self._prob_range()
        logger.info(
            "[Proactive] Registered evaluator (every 5 min, prob=%.4f–%.4f, "
            "technical max=%d/day).",
            lo, hi, self._max_daily(),
        )


def get_proactive_chat_id() -> int | None:
    """Determine which Telegram chat_id to send proactive messages to.

    Priority:
      1. proactive.chatId in config
      2. First entry in channels.telegram.allowedUsers
    """
    allowed = config.get_int_list("channels", "telegram", "allowedUsers")
    if not allowed:
        return None

    explicit = config.get_int("proactive", "chatId", default=0)
    if explicit:
        if explicit not in allowed:
            logger.error(
                "[Proactive] Refusing chatId=%s because it is not in Telegram allowedUsers",
                explicit,
            )
            return None
        return explicit

    return allowed[0]
