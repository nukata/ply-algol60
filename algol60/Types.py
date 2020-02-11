# H20.10.15/R02.02.11 by SUZUKI Hisao

import pprint

try:
    intern
except NameError:
    import sys
    intern = sys.intern         # for Python 3.*


class SemanticsError (Exception):
    pass


class Literal:
    def __init__(self, literal, lineno):
        # Treat `NL' within strings as a new line.
        self.literal = literal.replace("`NL'", "\n")
        self.lineno = lineno

class UnsignedInteger (Literal):
    def __repr__(self):
        return '%s_int' % self.literal

class UnsignedReal (Literal):
    def __repr__(self):
        return '%s_real' % self.literal

class String (Literal):
    def __repr__(self):
        return '%r' % self.literal


class Identifier:
    def __init__(self, name, lineno):
        self.name = intern(name)
        self.lineno = lineno
        self.quantity = None

    def __repr__(self):
        if self.quantity:
            return '%s@%x' % (self.name, id(self.quantity))
        else:
            return '%s' % self.name


class KeyWord (Identifier):
    pass

class Traversable:
    pass


def traverse(x, f):
    if f(x):
        if isinstance(x, list):
            for s in x:
                traverse(s, f)
        elif isinstance(x, Traversable):
            x.traverse(f)


class Quantity:
    def __init__(self, identifer_at_definition):
        self.identifier = identifer_at_definition

    def __repr__(self):
        return 'def %s@%x' % (self.identifier.name, id(self))


class LabelQuantity (Quantity):
    def __init__(self, identifer, target):
        Quantity.__init__(self, identifer)
        self.target = target


class SimpleVariableQuantity (Quantity):
    def __init__(self, identifier, type, is_own):
        Quantity.__init__(self, identifier)
        self.type = type
        self.is_own = is_own

    def __repr__(self):
        s = Quantity.__repr__(self) + ':%s' % self.type
        if self.is_own:
            s += '_own'
        return s


class ArrayQuantity (Quantity, Traversable):
    def __init__(self, identifier, bound_pair_list):
        Quantity.__init__(self, identifier)
        self.bound_pair_list = bound_pair_list
        self.type = None
        self.is_own = None

    def __repr__(self):
        s = Quantity.__repr__(self) + ':%s' % self.bound_pair_list
        if self.type:
            s += ':%s' %  self.type
        if self.is_own:
            s += '_own'
        return s

    def traverse(self, f):
        traverse(self.bound_pair_list, f)


class SwitchQuantity (Quantity, Traversable):
    def __init__(self, identifier, switch_list):
        Quantity.__init__(self, identifier)
        self.switch_list = switch_list

    def __repr__(self):
        return Quantity.__repr__(self) + ' switch-list %s' % self.switch_list

    def traverse(self, f):
        traverse(self.switch_list, f)


class FormalParameter (Quantity):
    def __init__(self, identifier):
        Quantity.__init__(self, identifier)
        self.by_value = False;
        self.quantity_class = None
        self.type = None

    def __repr__(self):
        s = '%s@%x' % (self.identifier.name, id(self))
        if self.by_value:
            s += ' by value'
        if self.quantity_class:
            if self.type:
                s += ' as %s %s' % (self.type, self.quantity_class.__name__)
            else:
                s += ' as ' + self.quantity_class.__name__
        return s


class ProcedureQuantity (Quantity, Traversable):
    def __init__(self, identifier, type, params, values, specs, body):
        Quantity.__init__(self, identifier)
        self.type = type
        # Enclose the bare body by a block (to delimit labels' scope).
        if not isinstance(body, Block):
            body = Block([], body)
            body.resolve_identifiers()
        # Do not resolve identifiers for quasi-procedures (e.g. (main)) where
        # params is None.  Otherwise resolve the procedure's name at least.
        if params is None:
            self.params = []
        else:
            # Define parameters by their specifications, constructing a
            # parameter dictionary.
            (self.params, dd) = define_params(params, values, specs)
            if identifier.name in dd:
                raise SemanticsError("same name as its param: %s at %d" %
                                     (identifier.name, identifier.lineno))
            # Add the procedure's name to the dictionary.
            dd[identifier.name] = self
            # Resolve each identifiers in the body with the dictionary.
            traverse(body, lambda x: resolve_identifier(dd, x))
        self.body = body

    def __repr__(self):
        return ('%s%s returns %s ' % (Quantity.__repr__(self),
                                      tuple(self.params),
                                      self.type) + 
                pprint.pformat(self.body))

    def traverse(self, f):
        traverse(self.body, f)


def define_params(params, values, specs):
    pp = []
    dd = {}
    for ident in params:
        q = FormalParameter(ident)
        pp.append(q)
        dd[ident.name] = q
    for v in values:
        if v.name in dd:
            dd[v.name].by_value = True
        else:
            raise SemanticsError("unknown name in 'value': %s at %d" %
                                 (v.name, v.lineno))
    for spec in specs:
        (specifier, identifier_list) = spec
        (quantity_class, type) = specifier
        for v in identifier_list:
            if v.name in dd:
                q = dd[v.name]
                if q.quantity_class is not None:
                    raise SemanticsError("double definition in parameter"
                                         " specification: %s at %d" %
                                         (v.name, v.lineno))
                q.quantity_class = quantity_class
                q.type = type
            else:
                raise SemanticsError("unknown name in parameter"
                                     " specification: %s at %d" %
                                     (v.name, v.lineno))
    return (pp, dd)


def resolve_identifier(dd, v):  # Resolve `v` with an identifier dictionary.
    if isinstance(v, Identifier):
        if v.quantity is None and v.name in dd:
            v.quantity = dd[v.name]
        return False
    else:
        return True             # Let it resolve the leaf nodes of `v`.


class LabelledStatement (Traversable):
    def __init__(self, label, statement):
        self.statement = statement
        self.label = LabelQuantity(label, self)

    def __repr__(self):
        return 'label %s: ' % self.label + pprint.pformat(self.statement)

    def traverse(self, f):
        traverse(self.statement, f)


def collect_labels(labels, x):  # Collect `x` into a list of labels.
    if isinstance(x, (Block, Operation, Quantity)):
        return False
    else:
        if isinstance(x, LabelledStatement):
            labels.append(x.label)
        return True             # Let it collect the leaf nodes of `x`.


class Block (Traversable):
    def __init__(self, declarations, statements):
        self.declarations = declarations
        self.statements = statements

    def resolve_identifiers(self):
        # Collect labels in the block and append them to the declations.
        labels = []
        traverse(self.statements, lambda x: collect_labels(labels, x))
        self.declarations.extend(labels)
        # Construct an identifier dictionary by the declarations.
        dd = {}
        for q in self.declarations:
            name = q.identifier.name
            if name in dd:
                raise SemanticsError("double definition in" 
                                     " declaration: %s at %d" %
                                     (name, q.identifier.lineno))
            dd[name] = q
        # Resolve each identifiers in the block with the dictionary.
        traverse(self, lambda x: resolve_identifier(dd, x))

    def __repr__(self):
        return 'block ' + pprint.pformat((self.declarations,
                                          self.statements)) + ' end-block'

    def traverse(self, f):
        traverse(self.declarations, f)
        traverse(self.statements, f)


class AssignmentStatement (Traversable):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return 'set %s := %s' % (self.lhs, self.rhs)

    def traverse(self, f):
        traverse(self.lhs, f)
        traverse(self.rhs, f)


class GoToStatement (Traversable):
    def __init__(self, desig, lineno):
        self.desig = desig
        self.lineno = lineno

    def __repr__(self):
        return 'goto ' + str(self.desig)

    def traverse(self, f):
        traverse(self.desig, f)


class DummyStatement:
    def __repr__(self):
        return 'pass'


class IfStatement (Traversable):
    def __init__(self, if_exp, if_lineno, then_s, then_lineno, 
                 else_s=None, else_lineno=None):
        self.if_exp = if_exp
        self.if_lineno = if_lineno
        self.then_s = then_s
        self.then_lineno = then_lineno
        self.else_s = else_s
        self.else_lineno = else_lineno

    def __repr__(self):
        if self.else_s:
            return 'if ' + pprint.pformat((self.if_exp,
                                           self.then_s, self.else_s))
        else:
            return 'if ' + pprint.pformat((self.if_exp, self.then_s))

    def traverse(self, f):
        traverse(self.if_exp, f)
        traverse(self.then_s, f)
        if self.else_s is not None:
            traverse(self.else_s, f)


class ForStatement (Traversable):
    def __init__(self, for_lineno, variable, for_list, do_lineno, statement):
        # Enclose the bare statement by a block (to delimit labels' scope).
        if not isinstance(statement, Block):
            statement = Block([], statement)
            statement.resolve_identifiers()
        self.for_lineno = for_lineno
        self.variable = variable
        self.for_list = for_list
        self.do_lineno = do_lineno
        self.statement = statement

    def __repr__(self):
        return 'for ' + pprint.pformat((self.variable, self.for_list,
                                        self.statement))

    def traverse(self, f):
        traverse(self.variable, f)
        traverse(self.for_list, f)
        traverse(self.statement, f)


class ProcedureStatement (Traversable):
    def __init__(self, fun, args):
        self.fun = fun
        self.args = args

    def __repr__(self):
        return 'call %s%s' % (self.fun, tuple(self.args))

    def traverse(self, f):
        traverse(self.fun, f)
        traverse(self.args, f)


class Operation (Traversable):
    def __init__(self, operator, *operands):
        self.op = operator
        self.arg = list(operands)

    def __repr__(self):
        return 'op:%s%s' % (self.op, tuple(self.arg))

    def traverse(self, f):
        traverse(self.op, f)
        traverse(self.arg, f)


class UnaryOperation (Operation):
    def __init__(self, op, arg):
        assert isinstance(op, KeyWord)
        Operation.__init__(self, op, arg)

    def __repr__(self):
        return '(%s %s)' % (self.op.name, self.arg[0])


class BinaryOperation (Operation):
    def __init__(self, left, op, right):
        assert isinstance(op, KeyWord)
        Operation.__init__(self, op, left, right)

    def __repr__(self):
        return '(%s %s %s)' % (self.arg[0], self.op.name, self.arg[1])


class IfExpression (Operation):
    def __init__(self, if_exp, if_lineno, then_exp, then_lineno,
                 else_exp, else_lineno):
        Operation.__init__(self, None, if_exp, then_exp, else_exp)
        self.if_lineno = if_lineno
        self.then_lineno = then_lineno
        self.else_lineno = else_lineno

    def __repr__(self):
        return '(%s ? %s : %s)' % (self.arg[0], self.arg[1], self.arg[2])


class FunctionDesignator (Operation):
    def __init__(self, fun, args):
        Operation.__init__(self, fun, *args)

    def __repr__(self):
        return 'call:%s%s' % (self.op, tuple(self.arg))


class SubscriptedValue (Operation):
    def __init__(self, array, subs):
        Operation.__init__(self, array, *subs)

    def __repr__(self):
        return '%s!%s' % (self.op, self.arg)


class SwitchDesignator (Operation):
    def __init__(self, array, subs):
        Operation.__init__(self, array, subs)

    def __repr__(self):
        return '%s$%s' % (self.op, self.arg)
