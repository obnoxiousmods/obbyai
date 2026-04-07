"""Calculator tool — safe math expression evaluator."""
import ast
import math
import operator

TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "Evaluate a mathematical expression accurately. Use this for any arithmetic, "
            "algebra, unit conversions, percentages, or numerical calculations. "
            "Supports: +, -, *, /, **, %, sqrt, sin, cos, tan, log, abs, round, floor, ceil, pi, e."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate. Example: '2 ** 10', 'sqrt(144)', 'sin(pi/2)'",
                }
            },
            "required": ["expression"],
        },
    },
}

_SAFE_NAMES = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "cbrt": lambda x: x ** (1/3),
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "log": math.log, "log2": math.log2, "log10": math.log10,
    "exp": math.exp, "pow": math.pow,
    "floor": math.floor, "ceil": math.ceil, "trunc": math.trunc,
    "factorial": math.factorial, "gcd": math.gcd,
    "pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf,
    "degrees": math.degrees, "radians": math.radians,
    "hypot": math.hypot,
}

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Name, ast.Load,
)


def _safe_eval(expr: str) -> float:
    tree = ast.parse(expr.strip(), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Disallowed expression node: {type(node).__name__}")
    return eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, _SAFE_NAMES)


async def run(expression: str) -> dict:
    try:
        result = _safe_eval(expression)
        # Format nicely
        if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
            formatted = str(int(result))
        elif isinstance(result, float):
            formatted = f"{result:.10g}"
        else:
            formatted = str(result)
        return {"expression": expression, "result": formatted}
    except ZeroDivisionError:
        return {"expression": expression, "error": "Division by zero"}
    except (ValueError, TypeError, OverflowError) as e:
        return {"expression": expression, "error": str(e)}
    except Exception as e:
        return {"expression": expression, "error": f"Could not evaluate: {e}"}
