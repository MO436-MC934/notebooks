import contextlib
import functools
import traceback

INDENT = 0
SUPRESS = False

def log(msg=None):
    """Prints a message at current indentation."""
    if msg is not None and not SUPRESS:
        print("|   " * INDENT + msg)


def _trace_indent(msg=None):
    """Print a message and increase indentation."""
    global INDENT
    log(msg)
    INDENT = INDENT + 1


def _trace_unindent(msg=None):
    """Decrease indentation and print a message."""
    global INDENT
    INDENT = INDENT - 1
    log(msg)


def log_calls(fn):
    """Decorator that shows when functions are invoked/returned."""
    def pp(v):
        """Pretty-print a value."""
        vtype = str(type(v))
        if "jax._src.lib.xla_bridge._JaxComputationBuilder" in vtype:
            return "<JaxComputationBuilder>"
        if "jaxlib.xla_extension.XlaOp" in vtype:
            return f"<XlaOp at 0x{id(v):x}>"
        if ("partial_eval.JaxprTracer" in vtype or
            "batching.BatchTracer" in vtype or
            "ad.JVPTracer" in vtype):
            return f"Tracer<{v.aval}>"
        if isinstance(v, tuple):
            return f"({pp_vals(v)})"

        return str(v)

    def pp_vals(args):
        return ", ".join(pp(arg) for arg in args)

    @functools.wraps(fn)
    def fn_wrapper(*args):
        """Wrapper of fn that shows the calls as they happen."""
        _trace_indent(f"| CALL {fn.__name__}({pp_vals(args)})")
        res = fn(*args)
        _trace_unindent(f"| RET  {fn.__name__} = {pp(res)}")
        return res

    return fn_wrapper


class ExpectNotImplemented:
    def __enter__(self):
        pass

    def __exit__(self, typ, value, tb):
        global INDENT
        INDENT = 0
        if typ is NotImplementedError:
            print(f"\nFound expected exception:")
            traceback.print_exc(limit=3)
            return True
        elif typ is None:
            assert False, "Expected NotImplementedError"
        else:
            return False

@contextlib.contextmanager
def SuppressCallLog():
    global SUPRESS
    SUPRESS = True
    try:
        yield None
    finally:
        SUPRESS = False
