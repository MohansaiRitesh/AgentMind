"""
╔══════════════════════════════════════════════════════════════════╗
║                    CONCEPT: TOOLS                                ║
║                                                                  ║
║  Tools are the agent's "hands" — ways to interact with           ║
║  the outside world beyond just generating text.                  ║
║                                                                  ║
║  A LangChain Tool has 3 required parts:                          ║
║  1. name:        string the LLM uses to call it                  ║
║  2. description: tells the LLM WHEN to use it (crucial!)         ║
║  3. func:        the actual Python function to run               ║
║                                                                  ║
║  BEST PRACTICE: Write descriptions that tell the LLM:            ║
║  - What the tool does                                            ║
║  - What kind of input it expects                                 ║
║  - When it's the RIGHT tool to use                               ║
║  - When NOT to use it                                            ║
║                                                                  ║
║  Good descriptions = smarter agents                              ║
╚══════════════════════════════════════════════════════════════════╝
"""

from langchain_core.tools import tool

# Try importing ddgs (new name) then fall back to duckduckgo_search (old name)
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None


# ─────────────────────────────────────────────────────────────────────
# CONCEPT: @tool decorator
# The @tool decorator from langchain_core converts a regular Python
# function into a LangChain tool. It automatically:
# - Uses the function name as the tool name
# - Uses the docstring as the tool description
# - Creates a Pydantic schema from the function's type hints
#   (this schema is what the LLM uses to know what args to pass)
# ─────────────────────────────────────────────────────────────────────

def create_search_tool():
    """Creates and returns the web search tool."""
    
    @tool
    def web_search(query: str, num_results: int = 5) -> str:
        """
        Search the web for current information using DuckDuckGo.
        
        Use this tool when you need to:
        - Find recent news, events, or developments
        - Look up facts that may have changed recently  
        - Research a topic you don't have full knowledge about
        - Verify information or find sources
        
        Do NOT use for math calculations or text transformations.
        
        Args:
            query: A clear, specific search query (2-8 words works best)
            num_results: How many results to return (default 5)
        
        Returns:
            Formatted search results with titles, snippets, and URLs
        """
        if DDGS is None:
            return f"Search unavailable (ddgs/duckduckgo_search not installed). Install with: pip install ddgs"

        try:
            results = []
            
            with DDGS() as ddgs:
                search_results = list(
                    ddgs.text(
                        query,
                        max_results=num_results,
                    )
                )
            
            if not search_results:
                return f"No results found for: {query}"
            
            for i, r in enumerate(search_results, 1):
                title = r.get("title", "No title")
                snippet = r.get("body", r.get("description", "No content"))
                url = r.get("href", r.get("url", "No URL"))
                results.append(
                    f"{i}. **{title}**\n"
                    f"   {snippet}\n"
                    f"   Source: {url}"
                )
            
            return "\n\n".join(results)
            
        except Exception as e:
            return f"Search error: {str(e)}. Try a different query."
    
    return web_search
