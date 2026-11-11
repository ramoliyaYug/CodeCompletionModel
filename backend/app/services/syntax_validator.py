import ast
from typing import Tuple, Optional


class SyntaxValidator:
    """
    Validates Python syntax of generated code using standard library `ast.parse`.
    NEVER executes the code. Strictly performs static AST parsing.
    """

    @staticmethod
    def validate_python_code(prompt: str, completion: str) -> Tuple[bool, Optional[str]]:
        """
        Validates whether prompt + completion (or completion standalone) is syntactically valid Python.
        
        Returns:
            (is_valid: bool, error_message: Optional[str])
        """
        full_code = prompt + completion

        # 1. Try parsing full concatenated code as an execution block
        try:
            ast.parse(full_code, mode="exec")
            return True, None
        except SyntaxError as e1:
            full_error = f"{e1.msg} at line {e1.lineno}, col {e1.offset}"

        # 2. Try parsing completion as an expression (e.g. `a + b`, `[x for x in items]`)
        try:
            ast.parse(completion.strip(), mode="eval")
            return True, None
        except SyntaxError:
            pass

        # 3. Try parsing completion as standalone statements
        try:
            ast.parse(completion.strip(), mode="exec")
            return True, None
        except SyntaxError:
            pass

        # 4. Check if prompt ended with an incomplete block that needs pass/dedent
        # e.g., prompt + completion + '\n    pass'
        try:
            ast.parse(full_code + "\n    pass", mode="exec")
            return True, None
        except SyntaxError:
            pass

        # If all parsing attempts fail, return the syntax error from the full code
        return False, full_error


syntax_validator = SyntaxValidator()
