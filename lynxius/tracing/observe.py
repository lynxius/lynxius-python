from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Avoiding circular imports
    import lynxius.evals.evaluator


class Span:
    def __init__(
        self,
        name: str,
        args: dict,
        kwargs: dict,
        start_time: datetime = None,
        end_time: datetime = None,
        output: dict = None,
    ):
        self.name = name
        self.args = args
        self.kwargs = kwargs
        self.start_time = start_time
        self.end_time = end_time
        self.output = output
        self.children: list[Span] = []


class Trace:
    def __init__(self, entry: Span):
        self.entry = entry
        self.stack: list[Span] = [entry]
        self.evals: list["lynxius.evals.evaluator.Evaluator"] = []

    def push(self, span: Span):
        self.stack[-1].children.append(span)
        self.stack.append(span)

    def pop(self):
        self.stack.pop()


trace_context: ContextVar[Trace] = ContextVar("stack_context", default=None)


def lynxius_observe(_func=None, name: str | None = None):
    def decorator_observe(func):
        @wraps(func)
        def wrapper_observe(*args, **kwargs):
            trace = trace_context.get()
            if not trace:
                # This is the first span in the trace.
                trace_context.set(Trace(entry=Span("entry", None, None)))
                trace = trace_context.get()

            span_name = name
            if span_name is None:
                span_name = func.__name__

            span = Span(
                name=span_name, args=args, kwargs=kwargs, start_time=datetime.now()
            )
            trace.push(span)

            # Run the function
            result = func(*args, **kwargs)

            span.end_time = datetime.now()
            span.output = result

            trace.pop()

            return result

        return wrapper_observe

    if _func is None:
        return decorator_observe
    else:
        return decorator_observe(_func)


def lynxius_finalize() -> Trace:
    trace = trace_context.get()
    trace_context.set(None)
    return trace


def _try_register_eval(eval: "lynxius.evals.evaluator.Evaluator"):
    """
    If there is an ongoing trace, this function attaches an evaluator to that trace.
    It's a nop if there's no trace currently being recorded.
    """

    trace = trace_context.get()
    if not trace:
        return

    trace.evals.append(eval)
