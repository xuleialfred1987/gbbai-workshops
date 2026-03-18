# Example Prompts for Testing send_to_teams Tool

## Basic Text Messages

1. "Send a message to Teams saying 'The deployment is complete!'"

2. "Send a notification to Teams with title 'System Alert' and message 'Database backup completed successfully'"

3. "Post a simple message to Teams: 'Meeting scheduled for 3 PM today'"

## Messages with Images

4. "Send a message to Teams with an Azure logo and text 'Azure deployment successful'"

5. "Post to Teams: 'New feature released!' with this image: https://picsum.photos/600/400"

## Messages with Charts

6. "Create a pie chart showing our Q4 revenue distribution: Sales $350k, Marketing $180k, Engineering $220k, Operations $150k, and send it to Teams"

7. "Send a bar chart to Teams showing monthly users: January 1200, February 1900, March 1500, April 2100, May 2400, June 2800"

8. "Post a line chart to Teams with weekly website traffic: Monday 420, Tuesday 550, Wednesday 380, Thursday 690, Friday 820, Saturday 750, Sunday 880"

9. "Send a doughnut chart to Teams showing browser usage: Chrome 65%, Firefox 15%, Safari 10%, Edge 10%"

10. "Create a pie chart with our department budgets and send to Teams: Sales $500k, Engineering $800k, Marketing $300k, HR $200k"

## Complex Reports

11. "Send a revenue report to Teams with a pie chart showing Q4 distribution across Sales, Marketing, Engineering, and Operations departments"

12. "Post our weekly traffic statistics to Teams with a line chart titled 'Website Visitors'"

13. "Send a notification to Teams with title 'Monthly Performance' and include a bar chart of our active user growth over the past 6 months"

14. "Create a Teams notification about browser statistics with a doughnut chart showing Chrome, Firefox, Safari, and Edge market share"

## Mixed Content

15. "Send a message to Teams titled 'Project Update' saying 'Phase 1 completed ahead of schedule' with a celebratory image"

16. "Post to Teams: 'Q4 Financial Summary' with a chart showing revenue vs expenses comparison for the last 6 months"

17. "Send a Teams message with title 'Analytics Report' containing a line chart of daily visitor trends for the past week"

## How the AI Will Process These

When you give these prompts to the AI assistant, it will:
1. Parse your request
2. Identify that you want to send something to Teams
3. Extract the message, title, and data
4. Structure the chart configuration if chart data is provided
5. Call the `send_to_teams` tool with appropriate parameters
6. Return confirmation or error message

## Notes

- The AI can intelligently extract data from natural language
- You don't need to specify exact parameter names
- The AI will choose appropriate chart types based on your data
- Colors are automatically assigned using a default palette
- You can mix and match any combination of title, message, image, and chart