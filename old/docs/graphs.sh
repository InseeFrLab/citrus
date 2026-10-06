
# All
# find . -type f -name "*.py" \
# -not -path "./.venv/*" \
# -print | xargs uv run code2flow --output docs/call_graph_all.png

# All but test functions
find . -type f -name "*.py" \
-not -path "./.venv/*" \
-not -path "./src/test/*" \
-print | xargs uv run code2flow --output docs/call_graph_all_but_test.png