"""
Calculator tool — demonstrates that tools can be pure Python functions,
not just API calls. LLMs are notoriously bad at arithmetic, so giving
them a calculator tool dramatically improves accuracy for math tasks.
"""

from langchain_core.tools import tool
import math
import ast
import operator


def create_calculator_tool():
    """Creates a safe math expression evaluator tool."""
    
    # Safe operators whitelist (no exec/eval of arbitrary code)
    SAFE_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.Mod: operator.mod,
    }
    SAFE_FUNCS = {
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "abs": abs,
        "round": round,
        "pi": math.pi,
        "e": math.e,
    }
    
    def safe_eval(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            op = SAFE_OPS.get(type(node.op))
            if op:
                return op(safe_eval(node.left), safe_eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            op = SAFE_OPS.get(type(node.op))
            if op:
                return op(safe_eval(node.operand))
        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name in SAFE_FUNCS:
                args = [safe_eval(a) for a in node.args]
                return SAFE_FUNCS[func_name](*args)
        elif isinstance(node, ast.Name):
            if node.id in SAFE_FUNCS:
                return SAFE_FUNCS[node.id]
        raise ValueError(f"Unsafe operation: {ast.dump(node)}")
    
    @tool
    def calculator(expression: str) -> str:
        """
        Evaluate a mathematical expression safely.
        
        Use this tool for:
        - Arithmetic: "2 + 2", "100 * 1.08 ** 10"
        - Math functions: "sqrt(144)", "log(1000)"
        - Constants: "pi * 5**2" (area of circle)
        
        Do NOT use for text processing or web searches.
        Always use this instead of trying to calculate in your head.
        
        Args:
            expression: A mathematical expression string
            
        Returns:
            The computed result as a string
        """
        try:
            tree = ast.parse(expression, mode='eval')
            result = safe_eval(tree.body)
            return f"{expression} = {result}"
        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception as e:
            return f"Calculation error: {str(e)}. Check expression syntax."
    
    return calculator


def create_summarizer_tool():
    """Creates a text summarization tool."""
    
    @tool
    def summarize_text(text: str, max_sentences: int = 3) -> str:
        """
        Summarize a long piece of text into key points.
        
        Use this tool when you have gathered long search results
        or content that needs to be condensed before analysis.
        
        Args:
            text: The text to summarize
            max_sentences: Target number of sentences in the summary
            
        Returns:
            A concise summary of the key points
        """
        # Simple extractive summarization:
        # Take first sentence + longest sentences as key points
        sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if len(s.strip()) > 30]
        
        if not sentences:
            return text[:500] + "..." if len(text) > 500 else text
        
        if len(sentences) <= max_sentences:
            return '. '.join(sentences) + '.'
        
        # Score sentences by length (simple heuristic: longer = more content)
        scored = sorted(enumerate(sentences), key=lambda x: len(x[1]), reverse=True)
        top_indices = sorted([i for i, _ in scored[:max_sentences]])
        summary = '. '.join(sentences[i] for i in top_indices) + '.'
        
        return summary
    
    return summarize_text
