# H20.10.29/R02.02.11 by SUZUKI Hisao

# XXX Type checks are incomplete.

from __future__ import print_function
import operator, sys
from .Types import *
from . import Prelude

DEBUG = False
#DEBUG = True

try: reduce
except NameError:
    from functools import reduce # for Python 3.*

try: intern
except NameError:
    intern = sys.intern         # for Python 3.*

try: raw_input
except NameError:
    raw_input = input           # for Python 3.*

try: INTEGERS = (int, long)
except NameError:
    INTEGERS = int              # for Python 3.*


BOOLEAN = intern('Boolean')
INTEGER = intern('integer')
PRINT = intern('print')
READ = intern('read')
REAL = intern('real')
STEP = intern('step')
WHILE = intern('while')
NATIVECALL = intern('_nativecall') # Unofficial: invoke Python function


class Frame:
    def __init__(self, proc, args, caller, interp, lineno):
        assert isinstance(proc, ProcedureQuantity), proc
        self.procedure = proc
        self.caller = caller
        self.level = proc.level
        params = proc.params
        self.vars = [None] * (len(params) + len(proc.locals))
        if len(args) != len(params):
            raise SemanticsError("%d arg(s) passed to '%s' (arity %d) at %d"
                                 % (len(args), proc.identifier.name,
                                    len(params), lineno))
        for (i, exp) in enumerate(args):
            p = params[i]
            if p.quantity_class is SimpleVariableQuantity:
                if p.by_value:
                    exp = interp.eval_expression(exp)
                    exp = coerce_to(p.type, exp, lineno)
            elif p.quantity_class is ArrayQuantity:
                assert isinstance(exp, Identifier)
                exp = interp.get_local_var(exp.quantity)
                assert isinstance(exp, ArrayValue)
                if p.by_value:
                    exp = exp.copy(p.type, lineno)
            elif p.quantity_class is SwitchQuantity:
                assert isinstance(exp, Identifier)
                exp = interp.get_local_var(exp.quantity)
                assert isinstance(exp, list)
            elif p.quantity_class is ProcedureQuantity:
                assert isinstance(exp, Identifier)
                exp = exp.quantity
                assert isinstance(exp, (ProcedureQuantity, FormalParameter))
            self.vars[i] = exp
        if DEBUG:
            print(" FRAME %s %s" % (self, self.vars))

    def __repr__(self):
        return "%s:%d@%x" % (self.procedure.identifier, self.level, id(self))


class CompoundStatement (Traversable):
    def __init__(self):
        self.next_statement = None

    def __repr__(self):
        r = []
        s = self
        while True:
            s = s.next_statement
            if s is None:
                break
            r.append(repr(s))
            if s.is_terminal:
                break
        return "begin " + "; ".join(r) + " end"
    
    def traverse(self, f):
        s = self
        while True:
            s = s.next_statement
            if s is None:
                break
            traverse(s, f)
            if s.is_terminal:
                break


def make_statements_threaded(x, home):
    """Link each statements in `x` with the `next_statement` attribute.
    The last statement in `x` is linked to `home`.
    """
    if x is None:
        return None
    elif isinstance(x, list):
        if len(x) == 1:
            return make_statements_threaded(x[0], home)
        c = CompoundStatement()        
        tail = __make_threaded(c, x)
        x = c
    else:
        __make_inner_threaded(x)
        tail = x
    tail.next_statement = home
    tail.is_terminal = True
    return x

def __make_threaded(prev, x):   # returns tail
    if isinstance(x, list):
        for s in x:
            prev = __make_threaded(prev, s)
        return prev
    else:
        __make_inner_threaded(x)
        prev.next_statement = x
        prev.is_terminal = False
        return x

def __make_inner_threaded(x):
    if isinstance(x, Block):
        x.statements = make_statements_threaded(x.statements, x)
    elif isinstance(x, IfStatement):
        x.then_s = make_statements_threaded(x.then_s, x)
        x.else_s = make_statements_threaded(x.else_s, x)
    elif isinstance(x, LabelledStatement):
        x.statement = make_statements_threaded(x.statement, x)


def resolve_locals(proc):
    assert isinstance(proc, ProcedureQuantity)
    level = proc.level
    for (i, var) in enumerate(proc.params):
        var.offset = i
        var.level = level
    args_len = len(proc.params)
    for (i, var) in enumerate(proc.locals):
        var.offset = i + args_len
        var.level = level


class GoToException (Exception):
    def __init__(self, label):
        self.label = label


class Interpreter:
    def __init__(self, parser):
        self.parser = parser

    def run(self, source_text):
        "Run `source_text` as a program in Algol."
        prelude = self.parser.parse(Prelude.PRELUDE)
        assert isinstance(prelude, Block)
        assert isinstance(prelude.statements[0], DummyStatement)
        s = program_tree = self.parser.parse(source_text)
        while isinstance(s, LabelledStatement):
            s = s.statement
        if isinstance(s, Block):
            s.resolve_identifiers()
        # Construct a block from the declarations and statements of PRELUDE.
        prelude.statements[0] = program_tree
        tree = Block(prelude.declarations, prelude.statements)
        tree.resolve_identifiers()
        main = ProcedureQuantity(Identifier("(main)", 1), type=None,
                                 params=None, values=None, specs=None,
                                 body=tree)
        self.max_level = 0
        self.owns = []
        traverse(main, lambda y: self.__transform(0, None, y))
        self.display = [None] * (self.max_level + 1)
        self.stack = []
        frame = Frame(main, [], None, self, 1)
        for q in self.owns: # Initialize each `own` variables in the frame.
            if isinstance(q, SimpleVariableQuantity):
                frame.vars[q.offset] = zero_value_for(q.type)
            else:
                assert isinstance(q, ArrayQuantity), q
                zero = zero_value_for(q.type)
                frame.vars[q.offset] = self.make_array(q, zero)
        self.in_new_line = True # True if a new line has just been printed.
        self.do_statement(frame)
        if not self.in_new_line:
            sys.stdout.write("\n")

    def stack_push(self, frame):
        """
        Push `frame` to `self.stack`,
        setting `self.display[level]` for the `level` of `frame`.
        """
        level = frame.level
        frame.old_display = self.display[level]
        self.display[level] = frame
        self.stack.append(frame)

    def stack_pop(self):
        """
        Pop and return `frame` from `self.stack`,
        recovering `self.display[level]` for the `level` of `frame`.
        """
        frame = self.stack.pop()
        self.display[frame.level] = frame.old_display
        del frame.old_display
        return frame

    def __transform(self, level, vars, x):
        """
        For a procedure `x`, give it an `level` (1, 2, ...) and link each
        statements within its body;
        aggregate each non-own variable quantities declared within it
        into its`locals` attribute; aggregate `own` variables into its `owns`.
        Give each labels within it a `procedure` attribute which refers to
        the enclosing proecedure for the lablel.
        Make the body of each `for` statements a quasi-procedure.
        """
        if isinstance(x, ProcedureQuantity):
            if self.max_level < level:
                self.max_level = level
            x.level = level
            locals = []
            traverse(x.body,
                     lambda y: self.__transform(level + 1, locals, y))
            x.locals = self.__filter_out_non_locals(locals, x)
            x.body = make_statements_threaded(x.body, x)
            if level == 0:
                # Treat the `own` variables as the locals of (main).
                x.locals += self.owns 
            resolve_locals(x)
            return False
        elif isinstance(x, Block):
            vars.extend(x.declarations)
            return True
        elif isinstance(x, ForStatement):
            proc = ProcedureQuantity(Identifier("(for)", x.for_lineno),
                                     type=None,
                                     params=None, values=None, specs=None,
                                     body=x.statement)
            del x.statement # Now it is an error to traverse its statement.
            self.__transform(level, None, proc)
            x.quasi_procedure = proc
            return False
        else:
            return isinstance(x, (list, LabelledStatement, IfStatement))


    def __filter_out_non_locals(self, locals, proc):
        local_vars = []
        for q in locals:
            if isinstance(q, LabelQuantity):
                q.procedure = proc
            elif isinstance(q, (SimpleVariableQuantity, ArrayQuantity)):
                if q.is_own:
                    self.owns.append(q)
                else:
                    local_vars.append(q)
            elif isinstance(q, SwitchQuantity):
                local_vars.append(q)
        return local_vars


    def do_statement(self, frame):
        "Execute the body of the procedure referred by `frame`."
        assert isinstance(frame, Frame), frame
        self.stack_push(frame)
        x = frame.procedure.body
        while True:
            try:
                if isinstance(x, LabelledStatement):
                    x = x.statement
                    continue
                elif isinstance(x, ProcedureStatement):
                    proc = x.fun.quantity
                    if isinstance(proc, ProcedureQuantity):
                        frame = Frame(proc, x.args, x, self, x.fun.lineno)
                        self.stack_push(frame)
                        x = x.fun.quantity.body
                        continue
                    else:
                        self.call_function(x.fun, x.args)
                elif isinstance(x, Block):
                    self.do_declarations(x.declarations)
                    x = x.statements
                    continue
                elif isinstance(x, GoToStatement):
                    j = self.eval_expression(x.desig)
                    raise GoToException(j)
                elif isinstance(x, IfStatement):
                    e = self.eval_expression(x.if_exp)
                    assert isinstance(e, bool)
                    if e:
                        x = x.then_s
                        continue
                    elif x.else_s is not None:
                        x = x.else_s
                        continue
                elif isinstance(x, AssignmentStatement):
                    # At first, evaluate the subscripts of LHS.
                    subscripts = []
                    for v in x.lhs:
                        s = self.eval_subscript(v)
                        subscripts.append(s)
                    rhs = self.eval_expression(x.rhs)
                    for (i, v) in enumerate(x.lhs):
                        self.do_assignment(v, subscripts[i], rhs)
                    #
                elif isinstance(x, ForStatement):
                    self.do_for_statement(x)
                else:
                    assert isinstance(x, (DummyStatement, CompoundStatement))
                #
                while x.is_terminal:
                    x = x.next_statement
                    if isinstance(x, ProcedureQuantity):
                        if DEBUG:
                            print(" RETURN", self.stack)
                        frame = self.stack_pop()
                        x = frame.caller
                        if x is None: # x will be None if no more statements.
                            return    # Return normally.
                        assert isinstance(x, ProcedureStatement)
                x = x.next_statement
            except GoToException as ex:
                j = ex.label
                assert isinstance(j, LabelQuantity)
                while True:
                    frame = self.stack[-1]
                    if frame.procedure is j.procedure:
                        break
                    self.stack_pop()
                    if frame.caller is None:
                        if DEBUG:
                            print(" EXIT TO", j, self.stack)
                        raise ex # Exit globally.
                if DEBUG:
                    print(" JUMP TO", j, self.stack)
                x = j.target


    def do_declarations(self, declarations):
        for q in declarations:
            if isinstance(q, ArrayQuantity):
                if not q.is_own:
                    self.set_local_var(q, self.make_array(q, None))
            elif isinstance(q, SwitchQuantity):
                sw = []
                for exp in q.switch_list:
                    e = self.eval_expression(exp)
                    assert isinstance(e, LabelQuantity)
                    sw.append(e)
                self.set_local_var(q, sw)

    def make_array(self, q, init_value):
        assert isinstance(q, ArrayQuantity)
        offsets = []
        lengths = []
        for pair in q.bound_pair_list:
            assert isinstance(pair, BinaryOperation)
            lower_bound = pair.arg[0]
            upper_bound = pair.arg[1]
            lb = self.eval_expression(lower_bound)
            ub = self.eval_expression(upper_bound)
            check_for_integer(lb, pair.op.lineno)
            check_for_integer(ub, pair.op.lineno)
            assert lb <= ub, pair.op.lineno
            offsets.append(lb)
            lengths.append(ub - lb + 1)
        return ArrayValue(q.type, offsets, lengths, init_value)

    def eval_subscript(self, v):
        if isinstance(v, SubscriptedValue):
            subs = [self.eval_expression(e) for e in v.arg]
            for s in subs:
                check_for_integer(s, v.op.lineno)
            return subs
        else:
            return None

    def do_assignment(self, v, subscript, rhs):
        "Assign RHS to the variable.  RHS is an evaluated value."
        if isinstance(v, Identifier):
            q = v.quantity
            if isinstance(q, ProcedureQuantity):
                # Store RHS to `frame.result` as the proecedure's return value.
                frame = self.display[q.level]
                if frame.procedure is not q:
                    raise SemanticsError("result of %s defined in %s at %d "
                                         % (v.name,
                                            frame.procedure.identifier.name,
                                            v.lineno))
                rhs = coerce_to(q.type, rhs, v.lineno)
                frame.result = rhs
                if DEBUG:
                    print(" RESULT OF %s := %s: %s" % (frame, rhs, type(rhs)))
            else:
                # Assign RHS to a simple variable.
                if (isinstance(q, SimpleVariableQuantity)
                    or (isinstance(q, FormalParameter)
                        and q.quantity_class is SimpleVariableQuantity
                        and q.by_value)):
                    rhs = coerce_to(q.type, rhs, v.lineno)
                    self.set_local_var(q, rhs)
                    if DEBUG:
                        print(" SET %s[%d] := %s: %s" % (self.display[q.level],
                                                         q.offset,
                                                         rhs, type(rhs)))
                elif (isinstance(q, FormalParameter)
                      and q.quantity_class is SimpleVariableQuantity
                      and not q.by_value):
                    v1 = self.get_local_var(q)
                    current_frame = self.stack_pop()
                    try:
                        try:
                            subs = self.eval_subscript(v1)
                            self.do_assignment(v1, subs, rhs)
                        except self.VariableExpected as e:
                            raise SemanticsError (
                                "variable expected: %s at %d: %s" %
                                (v.name, v.lineno, e.extra_arg))
                    finally:
                        self.stack_push(current_frame)
                else:
                    raise SemanticsError("variable expected: %s at %d: %s"
                                         % (v.name, v.lineno, q))
        elif isinstance(v, SubscriptedValue):
            # Assign RHS to an array element.
            q = v.op.quantity
            a = self.get_local_var(q)
            assert isinstance(a, ArrayValue)
            rhs = coerce_to(q.type, rhs, v.op.lineno)
            a.setitem(subscript, rhs, v.op.lineno)
            if DEBUG:
                print(" ASET %s[%d][%s] := %s: %s" % (self.display[q.level],
                                                      q.offset, subscript,
                                                      rhs, type(rhs)))
                print(" ARRAY:", a.vector, a.offsets, a.lengths)
        else:
            raise self.VariableExpected("variable expected: %s" % v, v)

    class VariableExpected (SemanticsError):
        def __init__(self, msg, extra_arg):
            SemanticsError.__init__(self, msg)
            self.extra_arg = extra_arg


    def do_for_statement(self, x):
        "Execute a for-statement `x`."
        var = x.variable
        proc = x.quasi_procedure
        for elem in x.for_list:
            if isinstance(elem, Operation):
                (op, arg) = (elem.op, elem.arg)
                if op.name is STEP:
                    (a, b, c) = arg
                    subscript = self.eval_subscript(var)
                    rhs = self.eval_expression(a)
                    self.do_assignment(var, subscript, rhs)
                    while True:
                        th = self.eval_expression(b)
                        vv = self.eval_expression(var)
                        cv = self.eval_expression(c)
                        d = vv - cv
                        if (th > 0 and d > 0) or (th < 0 and d < 0):
                            break
                        frame = Frame(proc, [], None, self, x.do_lineno)
                        self.do_statement(frame)
                        subscript = self.eval_subscript(var)
                        rhs = self.eval_expression(var) + th
                        self.do_assignment(var, subscript, rhs)
                    continue
                elif op.name is WHILE:
                    (e, f) = arg
                    while True:
                        subscript = self.eval_subscript(var)
                        rhs = self.eval_expression(e)
                        self.do_assignment(var, subscript, rhs)
                        t = self.eval_expression(f)
                        assert isinstance(t, bool), t
                        if not t:
                            break
                        frame = Frame(proc, [], None, self, x.do_lineno)
                        self.do_statement(frame)
                    continue
            subscript = self.eval_subscript(var)
            rhs = self.eval_expression(elem)
            self.do_assignment(var, subscript, rhs)
            frame = Frame(proc, [], None, self, x.do_lineno)
            self.do_statement(frame)


    def eval_expression(self, e):
        "Evaluate an expression `e`."
        if isinstance(e, (bool, int, float, str)):
            return e
        elif isinstance(e, Literal):
            if isinstance(e, UnsignedInteger):
                return int(e.literal)
            elif isinstance(e, UnsignedReal):
                return float(e.literal)
            else:
                assert isinstance(e, String)
                return e.literal
        elif isinstance(e, Identifier):
            q = e.quantity
            if (isinstance(q, ProcedureQuantity) or
                (isinstance(q, FormalParameter) and
                 q.quantity_class is ProcedureQuantity)):
                return self.call_function(e, [])
            # For a variable, take a bound value to its quantity.
            elif isinstance(q, (SimpleVariableQuantity)):
                return self.get_local_var(q)
            elif isinstance(q, FormalParameter):
                assert q.quantity_class is SimpleVariableQuantity, q
                v = self.get_local_var(q)
                if q.by_value:
                    return v
                else:
                    current_frame = self.stack_pop()
                    try:
                        return self.eval_expression(v)    
                    finally:
                        self.stack_push(current_frame)
            elif isinstance(q, LabelQuantity):
                return q
            else:
                assert q is None, q
                raise SemanticsError("undefined identifier: %s at %d" %
                                     (e.name, e.lineno))
        else:
            assert isinstance(e, Operation)
            (op, arg) = (e.op, e.arg)
            if isinstance(e, SubscriptedValue):
                subs = [self.eval_expression(e) for e in arg]
                q = op.quantity
                a = self.get_local_var(q)
                assert isinstance(a, ArrayValue)
                return a.getitem(subs, op.lineno)
            #
            elif isinstance(e, SwitchDesignator):
                subs = self.eval_expression(arg[0])
                q = op.quantity
                a = self.get_local_var(q)
                assert isinstance(a, list), a
                return a[subs - 1]
            #
            elif isinstance(e, IfExpression):
                (if_e, then_e, else_e) = arg
                exp = self.eval_expression(if_e)
                if exp is True:
                    return self.eval_expression(then_e)
                else:
                    assert exp is False
                    return self.eval_expression(else_e)
            #
            elif isinstance(e, UnaryOperation):
                operand = self.eval_expression(arg[0])
                op_name = op.name
                if op_name == '+':
                    return operand
                elif op_name == '-':
                    return - operand
                else:
                    assert op_name == 'not'
                    return not operand
            #
            elif isinstance(e, BinaryOperation):
                (left, right) = arg
                left = self.eval_expression(left)
                right = self.eval_expression(right)
                op_name = op.name
                if op_name == '+':
                    return left + right
                elif op_name == '-':
                    return left - right
                elif op_name == '*':
                    return left * right
                elif op_name == '/':
                    return float(left) / right
                elif op_name == 'div':
                    return left // right
                elif op_name == '=':
                    return left == right
                elif op_name == '/=':
                    return left != right
                elif op_name == '<':
                    return left < right
                elif op_name == '<=':
                    return left <= right
                elif op_name == '>':
                    return left > right
                elif op_name == '>=':
                    return left >= right
                elif op_name == 'and':
                    return left and right
                elif op_name == 'or':
                    return left or right
                elif op_name == 'impl':
                    return (not left) or right
                elif op_name == 'equiv':
                    assert isinstance(left, bool)
                    assert isinstance(right, bool)
                    return left == right
                else:
                    assert op_name == '**'
                    return left ** right
            else:
                assert isinstance(e, FunctionDesignator), e
                return self.call_function(op, arg)


    def __get_proc(self, q):
        assert isinstance(q, FormalParameter)
        assert q.quantity_class is ProcedureQuantity
        q = self.get_local_var(q)
        current_frame = self.stack_pop()
        try:
            if isinstance(q, ProcedureQuantity):
                return q
            else:
                return self.__get_proc(q)
        finally:
            self.stack_push(current_frame)

    def __call_with_frame(self, frame, q):
        assert isinstance(q, FormalParameter)
        assert q.quantity_class is ProcedureQuantity
        q = self.get_local_var(q)
        current_frame = self.stack_pop()
        try:
            if isinstance(q, ProcedureQuantity):
                assert frame.procedure is q
                frame.result = None
                self.do_statement(frame)
                return frame.result
            else:
                return self.__call_with_frame(frame, q)
        finally:
            self.stack_push(current_frame)

    def call_function(self, ident, args):
        "Invoke the function of `ident` with `args`, and return a result."
        assert isinstance(ident, Identifier)
        proc = ident.quantity
        if proc is None:
            name = ident.name
            if name is READ:
                self.do_read_procedure(args)
            elif name is PRINT:
                self.do_print_procedure(args)
            elif name is NATIVECALL:
                return self.call_native(args)
            else:
                raise SemanticsError("undefined procedure: %s at %d" %
                                     (name, ident.lineno))
        elif isinstance(proc, FormalParameter):
            q = self.__get_proc(proc)
            frame = Frame(q, args, None, self, ident.lineno)
            return self.__call_with_frame(frame, proc)
        else:
            frame = Frame(proc, args, None, self, ident.lineno)
            frame.result = None
            self.do_statement(frame)
            return frame.result

    def do_read_procedure(self, args):
        if not self.in_new_line:
            sys.stdout.write("\n")
        for v in args:
            subs = self.eval_subscript(v)
            if isinstance(v, Identifier):
                rhs = eval(raw_input("%s := " % v.name))
            else:
                assert isinstance(v, SubscriptedValue), v
                rhs = eval(raw_input("%s%s := " % (v.op.name, subs)))
            self.do_assignment(v, subs, rhs)
        self.in_new_line = True

    def do_print_procedure(self, args):
        if not self.in_new_line:
            sys.stdout.write("\n")
        is_first = True
        for e in args:
            if is_first:
                is_first = False
            else:
                sys.stdout.write(" ")
            value = self.eval_expression(e)
            sys.stdout.write(str(value))
        self.in_new_line = False

    def call_native(self, args):
        _a = [ self.eval_expression(e) for e in args ]
        assert isinstance(_a[0], str), _a
        _f = eval(_a[0])
        return _f(*_a[1:])


    def get_local_var(self, q):
        assert isinstance(q, Quantity)
        frame = self.display[q.level]
        return frame.vars[q.offset]

    def set_local_var(self, q, value):
        assert isinstance(q, Quantity)
        frame = self.display[q.level]
        frame.vars[q.offset] = value


class ArrayValue:
    def __init__(self, type, offsets, lengths, init_value):
        size = reduce(operator.mul, lengths)
        self.vector = [init_value] * size
        self.type = type
        self.offsets = offsets
        self.lengths = lengths

    def copy(self, type, lineno):
        r = ArrayValue(type, self.offsets, self.lengths, None)
        for (i, val) in enumerate(self.vector):
            r.vector[i] = coerce_to(type, val, lineno)
        return r

    def get_index(self, subscript, lineno):
        if len(subscript) != len(self.offsets):
            raise SemanticsError("dimensions differ (%d to %d) at %d" %
                                 (len(subscript), len(self.offsets), lineno))
        ii = list(map(operator.sub, subscript, self.offsets))
        for (i, val) in enumerate(ii):
            if not (0 <= val < self.lengths[i]):
                raise SemanticsError("index #%d (= %d) out of range at %d" %
                                     (i + 1, subscript[i], lineno))
        jj = list(map(operator.mul, ii[:-1], self.lengths[1:]))
        return reduce(operator.add, jj, ii[-1])

    def getitem(self, subscript, lineno):
        index = self.get_index(subscript, lineno)
        return self.vector[index]

    def setitem(self, subscript, value, lineno):
        index = self.get_index(subscript, lineno)
        self.vector[index] = value


def check_for_integer(exp, lineno):
    if not isinstance(exp, INTEGERS):
        raise SemanticsError("integer expected: %s at %d" % (exp, lineno))

def coerce_to(type, exp, lineno):
    if DEBUG:
        print(" COERCE:", type, exp, lineno)
    if type is None:
        raise SemanticsError("cannot coerce %s to void at %d" %
                             (exp, lineno))
    t = type.name
    if t is INTEGER:
        if isinstance(exp, bool):
            pass
        elif isinstance(exp, float):
            return int(round(exp))
        elif isinstance(exp, INTEGERS):
            return exp
    elif t is REAL:
        if isinstance(exp, bool):
            pass
        elif isinstance(exp, float):
            return exp
        elif isinstance(exp, INTEGERS):
            return float(exp)
    elif t is BOOLEAN:
        if isinstance(exp, bool):
            return exp
    raise SemanticsError("%s expected: %s at %d" % (t, exp, lineno))

def zero_value_for(type):
    t = type.name
    if t is INTEGER:
        return 0
    elif t is REAL:
        return  0.0
    else:
        assert t is BOOLEAN
        return False


if __name__ == '__main__':
    from .Parser import Parser
    import sys
    parser = Parser(write_tables=False, debug=False)
    interp = Interpreter(parser)
    for file_name in sys.argv[1:]:
        with open(file_name) as rf:
            source_text = rf.read()
        interp.run(source_text)
