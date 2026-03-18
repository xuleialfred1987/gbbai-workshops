#!/usr/bin/env python3
"""
Test script for the send_to_teams tool

This script tests the send_to_teams tool implementation.
You can use this to verify the tool works before integrating it into your application.
"""

import asyncio
import os
from tools import _send_to_teams_tool


def _ensure_webhook_configured() -> bool:
    if os.environ.get("TEAMS_WEBHOOK_URL"):
        return True

    print("TEAMS_WEBHOOK_URL is not configured. Skipping live Teams tool test.")
    return False


async def test_text_only():
    """Test 1: Text-Only Message"""
    print("\n" + "="*60)
    print("Test 1: Text-Only Message")
    print("="*60)
    
    args = {
        "message": "Hello from the AI Assistant! This is a text-only message.",
        "title": "AI Notification"
    }
    
    result = await _send_to_teams_tool(args)
    print(f"Result: {result.to_text()}")


async def test_minimal_request():
    """Test 2: Minimal Request (Only Required Field)"""
    print("\n" + "="*60)
    print("Test 2: Minimal Request")
    print("="*60)
    
    args = {
        "message": "Simple message without title or image"
    }
    
    result = await _send_to_teams_tool(args)
    print(f"Result: {result.to_text()}")


async def test_text_with_image():
    """Test 3: Text + Image URL"""
    print("\n" + "="*60)
    print("Test 3: Text + Image URL")
    print("="*60)
    
    args = {
        "message": "Check out this beautiful landscape!",
        "title": "Photo of the Day",
        "image_url": "https://picsum.photos/600/400"
    }
    
    result = await _send_to_teams_tool(args)
    print(f"Result: {result.to_text()}")


async def test_pie_chart():
    """Test 4: Pie Chart with chart_config"""
    print("\n" + "="*60)
    print("Test 4: Pie Chart")
    print("="*60)
    
    args = {
        "message": "Q4 2024 Revenue Distribution across all departments.",
        "title": "📊 Revenue Report",
        "chart_config": {
            "chart_type": "pie",
            "labels": ["Sales", "Marketing", "Engineering", "Operations"],
            "data": [350, 180, 220, 150],
            "chart_title": "Q4 Revenue by Department"
        }
    }
    
    result = await _send_to_teams_tool(args)
    print(f"Result: {result.to_text()}")


async def test_bar_chart():
    """Test 5: Bar Chart"""
    print("\n" + "="*60)
    print("Test 5: Bar Chart")
    print("="*60)
    
    args = {
        "message": "Monthly active users growth trend for the first half of 2024.",
        "title": "📈 User Growth",
        "chart_config": {
            "chart_type": "bar",
            "labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
            "data": [1200, 1900, 1500, 2100, 2400, 2800],
            "chart_title": "Monthly Active Users"
        }
    }
    
    result = await _send_to_teams_tool(args)
    print(f"Result: {result.to_text()}")


async def test_line_chart():
    """Test 6: Line Chart"""
    print("\n" + "="*60)
    print("Test 6: Line Chart")
    print("="*60)
    
    args = {
        "message": "Website traffic over the past week showing daily visitor counts.",
        "title": "🌐 Weekly Traffic",
        "chart_config": {
            "chart_type": "line",
            "labels": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            "data": [420, 550, 380, 690, 820, 750, 880],
            "chart_title": "Weekly Website Visitors"
        }
    }
    
    result = await _send_to_teams_tool(args)
    print(f"Result: {result.to_text()}")


async def test_doughnut_chart():
    """Test 7: Doughnut Chart"""
    print("\n" + "="*60)
    print("Test 7: Doughnut Chart")
    print("="*60)
    
    args = {
        "message": "Browser market share for Q4 2024.",
        "title": "🌍 Browser Distribution",
        "chart_config": {
            "chart_type": "doughnut",
            "labels": ["Chrome", "Firefox", "Safari", "Edge"],
            "data": [65, 15, 10, 10],
            "chart_title": "Browser Usage Stats",
            "colors": ["#4CAF50", "#2196F3", "#FFC107", "#E91E63"]
        }
    }
    
    result = await _send_to_teams_tool(args)
    print(f"Result: {result.to_text()}")


async def run_all_tests():
    """Run all tests"""
    if not _ensure_webhook_configured():
        return

    print("\n" + "="*60)
    print("SEND TO TEAMS TOOL TESTING SCRIPT")
    print("="*60)
    
    tests = [
        test_text_only,
        test_minimal_request,
        test_text_with_image,
        test_pie_chart,
        test_bar_chart,
        test_line_chart,
        test_doughnut_chart
    ]
    
    for i, test in enumerate(tests, 1):
        try:
            await test()
            await asyncio.sleep(0.5)  # Small delay between tests
        except Exception as e:
            print(f"❌ Test {i} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED!")
    print("="*60)
    print("\nCheck your Teams chat for the messages.")


async def main():
    """Main entry point"""
    import sys

    if not _ensure_webhook_configured():
        return
    
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        test_map = {
            '1': test_text_only,
            '2': test_minimal_request,
            '3': test_text_with_image,
            '4': test_pie_chart,
            '5': test_bar_chart,
            '6': test_line_chart,
            '7': test_doughnut_chart,
            'all': run_all_tests
        }
        
        if test_name in test_map:
            await test_map[test_name]()
        else:
            print(f"Unknown test: {test_name}")
            print("\nAvailable tests:")
            print("  1 - Text-only message")
            print("  2 - Minimal request")
            print("  3 - Text + image URL")
            print("  4 - Pie chart")
            print("  5 - Bar chart")
            print("  6 - Line chart")
            print("  7 - Doughnut chart")
            print("  all - Run all tests")
            print("\nUsage: python test_teams_tool.py [test_number|all]")
    else:
        await run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
