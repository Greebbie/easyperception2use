"""LLM middleware: natural language → perception commands via MiniMax, plus event interpretation."""

import asyncio
import json
import os
import re
import time
from typing import Optional

from openai import OpenAI

SYSTEM_PROMPT = """You are a vision system controller. A camera is running real-time object detection using YOLOv8.

You can detect these COCO classes:
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
traffic light, fire hydrant, stop sign, parking meter, bench,
bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe,
backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard,
sports ball, kite, baseball bat, baseball glove, skateboard, surfboard,
tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl,
banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza,
donut, cake, chair, couch, potted plant, bed, dining table, toilet,
tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven,
toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear,
hair drier, toothbrush

Given the user's question about what the camera sees, return a JSON object:
{
  "filter_classes": ["cow", "sheep"] or null,
  "response_template": "I can see {count} cows in the scene. {details}"
}

Rules:
- filter_classes: list of COCO class names to filter detection to, or null to detect all
- response_template: a natural language template. Use {count} for total count, {details} for per-region breakdown, {classes} for classes found
- If the user asks about animals, include relevant animal classes
- If the user asks a general question like "what do you see", set filter_classes to null
- Always respond in the same language the user uses
- Return ONLY the JSON object, no other text"""

EVENT_INTERPRET_PROMPT = """你是一个场景监控 AI 助手。你收到来自「场景事件中间层」的结构化事件数据。
所有检测、追踪、停留计算已由本地感知层完成，你不需要做任何视觉计算。

你的任务是：
1. 用中文自然语言描述正在发生什么
2. 如果有多个事件，关联它们形成连贯叙事
3. 评估综合威胁等级：normal / attention / warning / alert
4. 基于 SOP 规则建议具体动作：
   - 物品滞留 >60s → 确认是否需要清理或处理
   - 遗留物品（人离开但物品留下） → 通知安保 + 隔离区域
   - 静态物品被移除 → 核查清单，确认是否授权
   - 多个物品同时出现变化 → 密切关注
   - 复合事件（停留后移除） → 重点关注，建议回放

返回 JSON:
{
  "alert_text": "一句话中文告警摘要",
  "narrative": "2-3句中文完整叙事，描述发生了什么以及为什么值得关注",
  "threat_level": "normal" | "attention" | "warning" | "alert",
  "actions": ["中文建议动作1", "中文建议动作2"]
}

重要：所有字段必须用中文回复。只返回 JSON，不要返回其他内容。"""


# Mock responses for offline mode (no API key)
_MOCK_RESPONSES = {
    "zone_entry": {
        "alert_text": "新物品进入监控区域",
        "narrative": "检测到物品进入指定监控区域。系统已开始追踪该物品的位置和停留时间。",
        "threat_level": "normal",
        "actions": ["继续监控"],
    },
    "dwell_warning": {
        "alert_text": "物品滞留超过30秒",
        "narrative": "监控区域内的物品已停留超过30秒。如果该物品不应在此位置长时间停留，建议确认情况。",
        "threat_level": "attention",
        "actions": ["确认物品是否应在此位置", "如无人认领，准备进一步处理"],
    },
    "dwell_alert": {
        "alert_text": "物品滞留超过60秒，建议处理",
        "narrative": "监控区域内的物品已停留超过60秒，超出正常停留时间。该物品可能为遗留物或需要清理的物品。",
        "threat_level": "warning",
        "actions": ["派人确认物品归属", "如无人认领，隔离区域并通知安保"],
    },
    "dwell_critical": {
        "alert_text": "物品长时间滞留，需要立即处理",
        "narrative": "监控区域内的物品已停留超过120秒，远超正常范围。请立即安排人员核查。",
        "threat_level": "alert",
        "actions": ["立即派遣巡查", "隔离该区域", "通知安保负责人"],
    },
    "object_removed": {
        "alert_text": "物品被移除",
        "narrative": "监控区域内的物品已被移除。需确认此操作是否经过授权。",
        "threat_level": "attention",
        "actions": ["确认移除操作是否授权", "核查清单"],
    },
    "suspicious_removal": {
        "alert_text": "异常移除：物品长时间停留后被取走",
        "narrative": "一个长时间停留的物品突然被移除。该模式值得重点关注——物品先是长时间无人看管，随后被取走。建议回放录像确认。",
        "threat_level": "alert",
        "actions": ["回放录像确认操作人身份", "核查物品清单", "通知安保负责人"],
    },
}

_DEFAULT_MOCK = {
    "alert_text": "场景正常",
    "narrative": "当前监控区域无异常事件。系统持续监控中。",
    "threat_level": "normal",
    "actions": ["继续监控"],
}


class LLMMiddleware:
    """Translates natural language queries into perception commands and responses,
    and interprets structured events into natural language alerts."""

    MAX_INTERPRET_PER_MIN = 10

    def __init__(self) -> None:
        api_key = os.environ.get("DEMO_LLM_API_KEY", "")
        self._has_api_key = bool(api_key)
        if not api_key:
            print("[LLMMiddleware] WARNING: DEMO_LLM_API_KEY not set — using mock mode")
        self._client = OpenAI(
            api_key=api_key or "mock",
            base_url="https://api.minimax.chat/v1",
        )
        self._model = "minimax-m2.7"
        # Rate limiting for interpret_events
        self._interpret_timestamps: list[float] = []
        self._last_interpret_hash: str = ""

    # ── Scene Query (existing) ──

    async def process_query(
        self, user_text: str, perception_client
    ) -> dict:
        """Process a natural language query and return structured response."""
        try:
            llm_response = await self._call_llm(user_text)
        except Exception as e:
            return {
                "answer": f"LLM error: {e}",
                "scene_data": None,
                "actions_taken": [],
            }

        actions_taken = []

        filter_classes = llm_response.get("filter_classes")
        if filter_classes is not None:
            await perception_client.set_config("filter_classes", filter_classes)
            actions_taken.append(f"Filtering to: {', '.join(filter_classes)}")
            await asyncio.sleep(0.3)
        else:
            await perception_client.set_config("filter_classes", [])
            actions_taken.append("Detecting all classes")
            await asyncio.sleep(0.1)

        scene = await perception_client.get_latest()
        answer = self._format_response(llm_response, scene)

        return {
            "answer": answer,
            "scene_data": scene,
            "actions_taken": actions_taken,
        }

    # ── Event Interpretation (new) ──

    async def interpret_events(
        self, events_summary: dict, scene_summary: Optional[dict] = None,
    ) -> dict:
        """Interpret structured events via LLM (or mock if no API key).

        Args:
            events_summary: from EventAggregator.get_summary_for_llm()
            scene_summary: optional current scene dict

        Returns: {alert_text, narrative, threat_level, actions: []}
        """
        # Skip if no events
        if events_summary.get("total_events", 0) == 0:
            return _DEFAULT_MOCK.copy()

        # Dedup: skip if events haven't changed
        event_hash = _hash_events(events_summary)
        if event_hash == self._last_interpret_hash:
            return _DEFAULT_MOCK.copy()

        # Rate limit
        if not self._check_rate_limit():
            return self._mock_interpret(events_summary)

        self._last_interpret_hash = event_hash

        if not self._has_api_key:
            return self._mock_interpret(events_summary)

        try:
            return await self._llm_interpret(events_summary, scene_summary)
        except Exception as e:
            print(f"[LLMMiddleware] interpret_events error: {e}")
            return self._mock_interpret(events_summary)

    def _mock_interpret(self, events_summary: dict) -> dict:
        """Generate a mock LLM response based on highest severity event type."""
        events = events_summary.get("events", [])
        if not events:
            return _DEFAULT_MOCK.copy()

        # Find highest severity event
        severity_order = {"info": 0, "warning": 1, "alert": 2, "critical": 3}
        highest = max(events, key=lambda e: severity_order.get(e.get("severity", "info"), 0))

        event_type = highest.get("type", "")
        base = _MOCK_RESPONSES.get(event_type, _DEFAULT_MOCK).copy()

        # Enrich with actual event description
        desc = highest.get("description", "")
        if desc and base["narrative"]:
            base["narrative"] = f"{desc}。{base['narrative']}"

        return base

    async def _llm_interpret(
        self, events_summary: dict, scene_summary: Optional[dict],
    ) -> dict:
        """Call MiniMax LLM for event interpretation."""
        user_content = json.dumps(events_summary, ensure_ascii=False, indent=2)
        if scene_summary:
            user_content += "\n\n当前场景状态:\n" + json.dumps(
                scene_summary, ensure_ascii=False, indent=2
            )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": EVENT_INTERPRET_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.3,
                max_tokens=800,
            ),
        )
        content = response.choices[0].message.content.strip()
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [line for line in lines if not line.startswith("```")]
            content = "\n".join(lines).strip()

        try:
            result = json.loads(content)
            # Ensure required fields
            return {
                "alert_text": result.get("alert_text", ""),
                "narrative": result.get("narrative", ""),
                "threat_level": result.get("threat_level", "normal"),
                "actions": result.get("actions", []),
            }
        except json.JSONDecodeError:
            return self._mock_interpret(events_summary)

    def _check_rate_limit(self) -> bool:
        """Check if we can make another interpret call (max N per minute)."""
        now = time.time()
        self._interpret_timestamps = [
            t for t in self._interpret_timestamps if now - t < 60.0
        ]
        if len(self._interpret_timestamps) >= self.MAX_INTERPRET_PER_MIN:
            return False
        self._interpret_timestamps.append(now)
        return True

    # ── Context-Aware Chat ──

    async def context_chat(self, user_text: str, context: str) -> dict:
        """Answer user questions with full scene/event/dwell context."""
        if not self._has_api_key:
            return {"answer": f"[离线模式] 当前状态:\n{context}"}

        system = (
            "你是一个场景监控 AI 助手。你可以看到摄像头画面的实时分析结果。\n"
            "用户会用自然语言问你关于当前场景的问题。\n"
            "根据以下实时数据回答，用中文，简洁明了。\n\n"
            f"=== 实时数据 ===\n{context}\n=== 数据结束 ==="
        )

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_text},
                    ],
                    temperature=0.3,
                    max_tokens=500,
                ),
            )
            content = response.choices[0].message.content.strip()
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return {"answer": content}
        except Exception as e:
            return {"answer": f"分析失败: {e}"}

    # ── Watch Command (existing) ──

    async def process_watch_command(
        self, user_text: str, zone: Optional[dict],
        perception_client, watchpoint_monitor,
    ) -> dict:
        """Process a conversational watch/monitor command."""
        try:
            intent = await self._parse_watch_intent(user_text)
        except Exception as e:
            return {"answer": f"Sorry, I couldn't understand that: {e}", "action": "none"}

        action = intent.get("action", "none")
        target = intent.get("target_class", "person")
        label = intent.get("label", "Watched Area")
        reply = intent.get("reply", "")

        if action == "watch":
            if not zone:
                return {
                    "answer": "Please draw a rectangle on the video first to select the area you want me to monitor.",
                    "action": "need_zone",
                }
            watchpoint_monitor.clear_zones()
            zone_id = watchpoint_monitor.add_zone(
                x1=zone["x1"], y1=zone["y1"],
                x2=zone["x2"], y2=zone["y2"],
                target_class=target, label=label,
            )
            await perception_client.set_config("filter_classes", [])
            return {
                "answer": reply or f"Got it! I'm now monitoring the selected area for **{target}**. I'll alert you immediately when one is detected.",
                "action": "watching",
                "zone_id": zone_id,
                "target_class": target,
            }

        elif action == "stop":
            watchpoint_monitor.clear_zones()
            return {
                "answer": reply or "Monitoring stopped. Draw a new zone and tell me what to watch for whenever you're ready.",
                "action": "stopped",
            }

        elif action == "status":
            status = watchpoint_monitor.get_status()
            zones = status.get("zones", [])
            alerts = status.get("alert_count", 0)
            if zones:
                z = zones[0]
                return {
                    "answer": reply or f"Currently monitoring for **{z['target_class']}** at \"{z['label']}\". {alerts} alerts triggered so far.",
                    "action": "status",
                }
            return {
                "answer": reply or "No active monitoring. Draw a zone on the video and tell me what to watch.",
                "action": "status",
            }

        return {"answer": reply or "I can help you monitor areas in the camera feed. Draw a zone and tell me what to watch for!", "action": "none"}

    # ── Internal LLM calls ──

    async def _call_llm(self, user_text: str) -> dict:
        """Call MiniMax and parse the JSON response."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            temperature=0.3,
            max_tokens=500,
        ))
        content = response.choices[0].message.content.strip()
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [line for line in lines if not line.startswith("```")]
            content = "\n".join(lines).strip()
        return json.loads(content)

    def _format_response(self, llm_response: dict, scene: dict) -> str:
        """Fill the LLM's response template with actual scene data."""
        template = llm_response.get("response_template", "I see {count} objects. {details}")
        objects = scene.get("objects", [])
        scene_info = scene.get("scene", {})

        count = scene_info.get("object_count", len(objects))
        classes = scene_info.get("classes_present", [])

        region_summary = scene_info.get("region_summary", {})
        details_parts = []
        for region, cls_list in region_summary.items():
            readable_region = region.replace("_", " ")
            details_parts.append(f"{len(cls_list)} in {readable_region}")
        details = "; ".join(details_parts) if details_parts else "No objects detected"

        try:
            answer = template.format(
                count=count,
                details=details,
                classes=", ".join(classes) if classes else "none",
            )
        except KeyError:
            answer = template

        return answer

    async def _parse_watch_intent(self, user_text: str) -> dict:
        """Use LLM to parse monitoring intent from natural language."""
        watch_prompt = """You are an AI agent that controls a real-time camera perception system. You can see through the camera and take actions.

Your capabilities:
- Monitor a user-defined area for specific objects (person, car, dog, cat, cell phone, bottle, etc.)
- Alert the user when a target object enters the area
- Stop monitoring on request

The user may speak casually. Understand their intent and respond naturally.

Return a JSON object:
{
  "action": "watch" | "stop" | "status" | "none",
  "target_class": "person" | "car" | "dog" | "cat" | "cell phone" | "bottle" | etc,
  "label": "a short name for the zone",
  "reply": "your natural conversational reply in the user's language"
}

Rules:
- "watch": user wants you to monitor/watch/guard/detect something (e.g. "keep an eye on this", "tell me if someone comes", "monitor this area")
- "stop": user wants to stop (e.g. "stop", "that's enough", "cancel")
- "status": user asks what's happening (e.g. "anything yet?", "what's going on?")
- "none": just chatting, greeting, or asking general questions — reply naturally
- target_class must be a valid COCO detection class. Default is "person"
- If user mentions "hand", use "person" (hands are detected as part of a person)
- Reply conversationally, as if you're a capable AI assistant
- Return ONLY the JSON object"""

        if not self._has_api_key:
            return self._fallback_watch_intent(user_text)

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": watch_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0.3,
            max_tokens=1000,
        ))
        content = response.choices[0].message.content.strip()
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [line for line in lines if not line.startswith("```")]
            content = "\n".join(lines).strip()

        if not content:
            return self._fallback_watch_intent(user_text)

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return self._fallback_watch_intent(user_text)

    @staticmethod
    def _fallback_watch_intent(text: str) -> dict:
        """Keyword-based fallback when LLM fails to return valid JSON."""
        lower = text.lower()
        if any(w in lower for w in ["stop", "cancel", "quit", "enough", "停"]):
            return {"action": "stop", "target_class": "person", "label": "", "reply": ""}
        if any(w in lower for w in ["status", "what are you", "checking", "状态"]):
            return {"action": "status", "target_class": "person", "label": "", "reply": ""}
        if any(w in lower for w in ["watch", "monitor", "alert", "detect", "trigger", "盯", "监控", "提醒", "看"]):
            target = "person"
            if "car" in lower: target = "car"
            elif "dog" in lower: target = "dog"
            elif "cat" in lower: target = "cat"
            elif "phone" in lower: target = "cell phone"
            elif "bottle" in lower: target = "bottle"
            elif "laptop" in lower: target = "laptop"
            elif "cup" in lower: target = "cup"
            return {"action": "watch", "target_class": target, "label": "Watched Area", "reply": ""}
        return {"action": "none", "target_class": "person", "label": "", "reply": ""}


def _hash_events(events_summary: dict) -> str:
    """Create a simple hash of event types+descriptions for dedup."""
    events = events_summary.get("events", [])
    parts = [f"{e.get('type', '')}:{e.get('description', '')}" for e in events]
    return "|".join(parts)
