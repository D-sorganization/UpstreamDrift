import ast
from pathlib import Path

src_dir = Path("src")


class PrintCounter(ast.NodeVisitor):
    def __init__(self):
        self.count = 0
        self.files_with_print = []
        self.current_file = None

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            self.count += 1
            if self.current_file not in self.files_with_print:
                self.files_with_print.append(self.current_file)
        self.generic_visit(node)


counter = PrintCounter()

for p in src_dir.rglob("*.py"):
    try:
        content = p.read_text(encoding="utf-8")
        tree = ast.parse(content)
        counter.current_file = p
        counter.visit(tree)
    except Exception:
        pass

for _f in counter.files_with_print:
    pass
