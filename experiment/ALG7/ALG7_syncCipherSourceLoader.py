"""Load unmodified student cipher definitions, omitting offline WAV harness utilities.
Do not run side-effect tests in original source; turn off Numba disk cache only.
"""
import ast
from functools import lru_cache
from pathlib import Path

@lru_cache(maxsize=8)
def load_source(path):
    path=Path(path).resolve()
    tree=ast.parse(path.read_text(encoding='utf-8'),str(path))
    kept=[]
    for node in tree.body:
        if isinstance(node,ast.Import):
            names=[a.name for a in node.names]
            if 'script.utils' in names or 'matplotlib.pyplot' in names or (path.name=='script_wavelet.py' and 'pywt' in names):continue
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call):
            f=node.value.func
            if isinstance(f,ast.Attribute) and f.attr=='reload':continue
        if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='me' for x in node.targets):continue
        if isinstance(node,ast.If) and isinstance(node.test,ast.Compare) and isinstance(node.test.left,ast.Name) and node.test.left.id=='__name__':continue
        kept.append(node)
    tree.body=kept
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                if isinstance(dec,ast.Call) and isinstance(dec.func,ast.Name) and dec.func.id=='njit':
                    for kw in dec.keywords:
                        if kw.arg=='cache':kw.value=ast.Constant(False)
    ast.fix_missing_locations(tree)
    ns={'__file__':str(path),'__name__':'_nears_loaded_'+path.stem}
    exec(compile(tree,str(path),'exec'),ns)
    return ns
