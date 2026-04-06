"""End-to-end Playwright tests for the three-panel demo dashboard."""

import asyncio
import json
import time

import pytest
from playwright.sync_api import sync_playwright, Page, expect

BASE_URL = "http://localhost:8080"
NOPROXY = "localhost"


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def page(browser):
    context = browser.new_context(
        viewport={"width": 1400, "height": 900},
        # Bypass proxy for localhost
        proxy=None,
    )
    page = context.new_page()
    yield page
    context.close()


class TestDashboardLayout:
    """Test the three-panel layout loads correctly."""

    def test_page_loads(self, page: Page):
        page.goto(BASE_URL)
        expect(page).to_have_title("EasyPerception — 场景智能中间层")

    def test_three_panels_visible(self, page: Page):
        page.goto(BASE_URL)
        camera = page.locator(".camera-panel")
        expect(camera).to_be_visible()
        events = page.locator(".events-panel")
        expect(events).to_be_visible()
        llm = page.locator(".llm-panel")
        expect(llm).to_be_visible()

    def test_header_present(self, page: Page):
        page.goto(BASE_URL)
        expect(page.locator("h1")).to_have_text("EasyPerception")
        expect(page.locator(".subtitle")).to_have_text("场景智能中间层")

    def test_video_feed_loads(self, page: Page):
        page.goto(BASE_URL)
        video = page.locator("#videoFeed")
        expect(video).to_be_visible()
        # Check that image src is the MJPEG stream
        expect(video).to_have_attribute("src", "/stream")

    def test_draw_hint_visible(self, page: Page):
        page.goto(BASE_URL)
        hint = page.locator("#drawHint")
        expect(hint).to_be_visible()
        expect(hint).to_contain_text("拖拽框选")

    def test_stats_row_visible(self, page: Page):
        page.goto(BASE_URL)
        expect(page.locator("#statEvents")).to_be_visible()
        expect(page.locator("#statTracks")).to_be_visible()
        expect(page.locator("#statAlerts")).to_be_visible()
        expect(page.locator("#statDwell")).to_be_visible()

    def test_llm_sections_visible(self, page: Page):
        page.goto(BASE_URL)
        expect(page.locator("#llmAlertBox")).to_be_visible()
        expect(page.locator("#llmNarrative")).to_be_visible()
        expect(page.locator("#llmActions")).to_be_visible()

    def test_llm_chat_input(self, page: Page):
        page.goto(BASE_URL)
        input_el = page.locator("#llmInput")
        expect(input_el).to_be_visible()
        # Check placeholder contains Chinese text
        placeholder = input_el.get_attribute("placeholder")
        assert "发生" in placeholder or "异常" in placeholder or "问" in placeholder

    def test_footer_visible(self, page: Page):
        page.goto(BASE_URL)
        expect(page.locator(".footer")).to_be_visible()


class TestWebSocketConnection:
    """Test WebSocket connects and receives data."""

    def test_connection_status_connected(self, page: Page):
        page.goto(BASE_URL)
        # Wait for WebSocket to connect (should update status)
        page.wait_for_timeout(3000)
        status_dot = page.locator(".header-status .status-dot")
        expect(status_dot).to_have_class("status-dot connected")

    def test_stats_update_from_pipeline(self, page: Page):
        page.goto(BASE_URL)
        page.wait_for_timeout(3000)
        # Tracks should update (pipeline is running)
        tracks = page.locator("#statTracks")
        text = tracks.inner_text()
        # Should show a number (might be 0 or more)
        assert text.isdigit()


class TestZoneDrawingAndMonitoring:
    """Test drawing a zone and starting monitoring."""

    def test_draw_zone_auto_starts_monitoring(self, page: Page):
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        # Draw a zone — should auto-start monitoring
        canvas = page.locator("#zoneCanvas")
        box = canvas.bounding_box()
        if box:
            page.mouse.move(box["x"] + box["width"] * 0.1, box["y"] + box["height"] * 0.1)
            page.mouse.down()
            page.mouse.move(box["x"] + box["width"] * 0.9, box["y"] + box["height"] * 0.9)
            page.mouse.up()

        page.wait_for_timeout(1000)

        # Zone status should show "监控中"
        zone_status = page.locator("#zoneStatus")
        expect(zone_status).to_be_visible()

        # Draw hint should be hidden
        expect(page.locator("#drawHint")).to_be_hidden()

    def test_monitoring_generates_events(self, page: Page):
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        # Draw full-area zone (auto-start)
        canvas = page.locator("#zoneCanvas")
        box = canvas.bounding_box()
        if box:
            page.mouse.move(box["x"] + 10, box["y"] + 10)
            page.mouse.down()
            page.mouse.move(box["x"] + box["width"] - 10, box["y"] + box["height"] - 10)
            page.mouse.up()

        page.wait_for_timeout(5000)

        event_cards = page.locator(".event-card")
        assert event_cards.count() > 0, "Expected events after drawing zone"

    def test_clear_zone_stops_monitoring(self, page: Page):
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        canvas = page.locator("#zoneCanvas")
        box = canvas.bounding_box()
        if box:
            page.mouse.move(box["x"] + 10, box["y"] + 10)
            page.mouse.down()
            page.mouse.move(box["x"] + box["width"] - 10, box["y"] + box["height"] - 10)
            page.mouse.up()

        page.wait_for_timeout(1000)

        # Click the clear button
        page.click(".zone-clear-btn")
        page.wait_for_timeout(500)

        # Draw hint should reappear
        expect(page.locator("#drawHint")).to_be_visible()
        expect(page.locator("#zoneStatus")).to_be_hidden()


class TestDwellTimerDisplay:
    """Test that dwell timers appear when objects are in zone."""

    def test_dwell_timer_appears(self, page: Page):
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        # Draw full-area zone (auto-start)
        canvas = page.locator("#zoneCanvas")
        box = canvas.bounding_box()
        if box:
            page.mouse.move(box["x"] + 10, box["y"] + 10)
            page.mouse.down()
            page.mouse.move(box["x"] + box["width"] - 10, box["y"] + box["height"] - 10)
            page.mouse.up()

        page.wait_for_timeout(5000)

        dwell_section = page.locator("#dwellSection")
        if dwell_section.is_visible():
            timers = page.locator(".dwell-timer")
            assert timers.count() > 0


class TestAPIEndpoints:
    """Test the API endpoints directly."""

    def test_scene_endpoint(self, page: Page):
        page.goto(BASE_URL)
        response = page.request.get(f"{BASE_URL}/api/scene")
        assert response.status == 200
        data = response.json()
        assert "objects" in data
        assert "scene" in data

    def test_health_endpoint(self, page: Page):
        page.goto(BASE_URL)
        response = page.request.get(f"{BASE_URL}/api/health")
        assert response.status == 200
        data = response.json()
        assert data["connected"] is True
        assert data["status"] == "ok"

    def test_events_recent_endpoint(self, page: Page):
        page.goto(BASE_URL)
        response = page.request.get(f"{BASE_URL}/api/events/recent")
        assert response.status == 200
        data = response.json()
        assert "events" in data

    def test_zone_setup_endpoint(self, page: Page):
        page.goto(BASE_URL)
        response = page.request.post(
            f"{BASE_URL}/api/zone/setup",
            data=json.dumps({
                "zone": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
                "target_class": "bottle",
                "label": "Test Zone",
            }),
            headers={"Content-Type": "application/json"},
        )
        assert response.status == 200
        data = response.json()
        assert "zone_id" in data
        assert data["status"] == "monitoring"

        # Clean up
        page.request.post(
            f"{BASE_URL}/api/watchpoint",
            data=json.dumps({"action": "clear"}),
            headers={"Content-Type": "application/json"},
        )

    def test_watchpoint_status_endpoint(self, page: Page):
        page.goto(BASE_URL)
        response = page.request.get(f"{BASE_URL}/api/watchpoint/status")
        assert response.status == 200
        data = response.json()
        assert "zones" in data
        assert "alert_count" in data


class TestScreenshot:
    """Capture screenshots for visual verification."""

    def test_capture_initial_state(self, page: Page):
        page.goto(BASE_URL)
        page.wait_for_timeout(3000)
        page.screenshot(path="demo_screenshot_initial.png", full_page=False)

    def test_capture_monitoring_state(self, page: Page):
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        # Draw zone (auto-start)
        canvas = page.locator("#zoneCanvas")
        box = canvas.bounding_box()
        if box:
            page.mouse.move(box["x"] + 10, box["y"] + 10)
            page.mouse.down()
            page.mouse.move(box["x"] + box["width"] - 10, box["y"] + box["height"] - 10)
            page.mouse.up()

        page.wait_for_timeout(8000)

        page.screenshot(path="demo_screenshot_monitoring.png", full_page=False)
