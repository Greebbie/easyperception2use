"""LLM middleware: natural language → perception commands via MiniMax 2.7."""

import asyncio
import json
import os
import re
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


class LLMMiddleware:
    """Translates natural language queries into perception commands and responses."""

    def __init__(self):
        api_key = os.environ.get("DEMO_LLM_API_KEY", "")
        if not api_key:
            print("[LLMMiddleware] WARNING: DEMO_LLM_API_KEY not set")
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://api.minimax.chat/v1",
        )
        self._model = "minimax-m2.7"

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

    async def _call_llm(self, user_text: str) -> dict:
        """Call MiniMax 2.7 and parse the JSON response."""
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
        # Strip <think>...</think> tags from reasoning models
        import re
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            content = "\n".join(lines).strip()
        return json.loads(content)

    def _format_response(self, llm_response: dict, scene: dict) -> str:
        """Fill the LLM's response template with actual scene data."""
        template = llm_response.get("response_template", "I see {count} objects. {details}")
        objects = scene.get("objects", [])
        scene_info = scene.get("scene", {})

        count = scene_info.get("object_count", len(objects))
        classes = scene_info.get("classes_present", [])

        # Build per-region details
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

    async def process_watch_command(
        self, user_text: str, zone: Optional[dict],
        perception_client, watchpoint_monitor,
    ) -> dict:
        """Process a conversational watch/monitor command."""
        # Determine intent via LLM
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
            lines = [l for l in lines if not l.startswith("```")]
            content = "\n".join(lines).strip()

        # If LLM returned empty (thinking consumed all tokens), use keyword fallback
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
            return {"action": "watch", "target_class": target, "label": "Watched Area", "reply": ""}
        return {"action": "none", "target_class": "person", "label": "", "reply": ""}
