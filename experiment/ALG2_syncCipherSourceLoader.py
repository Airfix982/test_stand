"""Loads original student scripts unchanged, skipping their offline WAV harness imports.
The algorithm definitions themselves are executed unmodified; only Numba disk cache disabled.
"""
import ast
from pathlib import Path
from functools import lru_cache

@lru_cache(maxsize=8)
def load_source(path):
    path=Path(path).resolve()
    tree=ast.parse(path.read_text(encoding='utf-8'),str(path))
    kept=[]
    for n in tree.body:
        if isinstance(n,ast.Import) and any(x.name=='script.utils' or x.name=='matplotlib.pyplot' for x in n.names):continue
        if isinstance(n,ast.Import) and path.name=='script_wavelet.py' and any(x.name=='pywt' for x in n.names):continue
        if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='reload':continue
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='me' for t in n.targets):continue
        if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__':continue
        kept.append(n)
    tree.body=kept
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                if isinstance(dec,ast.Call) and isinstance(dec.func,ast.Name) and dec.func.id=='njit':
                    for kw in dec.keywords:
                        if kw.arg=='cache':kw.value=ast.Constant(False)
    ast.fix_missing_locations(tree)
    ns={'__file__':str(path),'__name__':'_nears_orig_'+path.stem}
    import __future__
    exec(compile(tree,str(path),'exec', flags=__future__.annotations.compiler_flag),ns)
    return ns
