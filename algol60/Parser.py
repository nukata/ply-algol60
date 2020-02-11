# H20.10.09/R02.02.11 by SUZUKI Hisao

from .ply import yacc
from .Lexer import Lexer, SyntaxError
from .Types import *

class Parser:
    tokens = Lexer.tokens

    def __init__(self, **yacc_args):
        self.lexer = None
        self.yacc_args = yacc_args

    def parse(self, source_text):
        if self.lexer is None:
            self.lexer = Lexer()
            self.parser = yacc.yacc(module=self, **self.yacc_args)
        return self.parser.parse(source_text, lexer=self.lexer)

    def p_error(self, t):
        if t is None:
            raise SyntaxError("unexpected token", self.lexer, None)
        else:
            raise SyntaxError("unexpected token", 
                              self.lexer, t.value, t.lineno, t.lexpos)


    # 4.1 Compound statements and blocks
    def p_program(self, p):
        '''program : block
                   | compound_statement'''
        p[0] = p[1]


    def p_block_1(self, p):
        'block : unlabelled_block'
        p[0] = p[1]

    def p_block_2(self, p):
        'block : label COLON block'
        p[0] = LabelledStatement(p[1], p[3])


    def p_compound_statement_1(self, p):
        'compound_statement : unlabelled_compound'
        p[0] = p[1]

    def p_compound_statement_2(self, p):
        'compound_statement : label COLON compound_statement'
        p[0] = LabelledStatement(p[1], p[3])


    def p_unlabelled_block(self, p):
        'unlabelled_block : block_head semicolon compound_tail'
        p[0] = Block(p[1], p[3])

    def p_unlabelled_compound(self, p):
        'unlabelled_compound : begin compound_tail'
        p[0] = p[2]


    def p_bock_head_1(self, p):
        'block_head : begin declaration'
        p[0] = p[2]

    def p_bock_head_2(self, p):
        'block_head : block_head semicolon declaration'
        p[0] = p[1] + p[3]


    def p_compound_tail_1(self, p):
        'compound_tail : statement END'
        p[0] = [p[1]]

    def p_compound_tail_2(self, p):
        'compound_tail : statement semicolon compound_tail'
        p[0] = [p[1]] + p[3]


    def p_statement(self, p):
        '''statement : unconditional_statement
                     | conditional_statement
                     | for_statement'''
        p[0] = p[1]


    def p_unconditional_statement_1(self, p):
        '''unconditional_statement : basic_statement
                                   | compound_statement'''
        p[0] = p[1]

    def p_unconditional_statement_2(self, p):
        'unconditional_statement : block'
        p[0] = p[1]
        s = p[1]
        while isinstance(s, LabelledStatement):
            s = s.statement
        assert isinstance(s, Block)
        s.resolve_identifiers()


    def p_basic_statement_1(self, p):
        'basic_statement : unlabelled_basic_statement'
        p[0] = p[1]

    def p_basic_statement_2(self, p):
        'basic_statement : label COLON basic_statement'
        p[0] = LabelledStatement(p[1], p[3])


    def p_unlabelled_basic_statement(self, p):
        '''unlabelled_basic_statement : assignment_statement
                                      | goto_statement
                                      | procedure_statement
                                      | dummy_statement'''
        p[0] = p[1]


    # 1.1 (Formalism for syntactic description)
    def p_empty(self, p):
        'empty :'
        pass

    # 2.3 (Delimiters)
    def p_semicolon(self, p):
        '''semicolon : SEMICOLON
                     | semicolon COMMENT'''
        pass

    def p_begin(self, p):
        '''begin : BEGIN
                 | begin COMMENT'''
        pass

    # 2.5 Numbers
    def p_unsigned_number_i(self, p):
        'unsigned_number : UNSIGNEDINTEGER'
        p[0] = UnsignedInteger(p[1], p.lineno(1))

    def p_unsigned_number_r(self, p):
        'unsigned_number : UNSIGNEDREAL'
        p[0] = UnsignedReal(p[1], p.lineno(1))

    # 2.6 Strings
    def p_string_1(self, p):
        'string : CLOSEDSTRING'
        p[0] = String(p[1], p.lineno(1))

    def p_string_2(self, p):
        'string : CLOSEDSTRING string'
        p[2].literal = p[1] + p[2].literal
        p[0] = p[2]


    # 3 Expressions
    def p_expression(self, p):
        '''expression : arith_expression
                      | bool_expression
                      | desig_expression'''
        p[0] = p[1]


    # 3.1 Variables
    def p_variable_identifier(self, p):
        'variable_identifier : IDENTIFIER'
        p[0] = Identifier(p[1], p.lineno(1))

    def p_simple_variable(self, p):
        'simple_variable : variable_identifier'
        p[0] = p[1]

    def p_subscript_expression(self, p):
        'subscript_expression : arith_expression'
        p[0] = p[1]


    def p_subscript_list_1(self, p):
        'subscript_list : subscript_expression'
        p[0] = [p[1]]

    def p_subscript_list_2(self, p):
        'subscript_list : subscript_list COMMA subscript_expression'
        p[0] = p[1] + [p[3]]


    def p_array_identifier(self, p):
        'array_identifier : IDENTIFIER'
        p[0] = Identifier(p[1], p.lineno(1))

    def p_subscripted_value(self, p):
        'subscripted_value : array_identifier LBRACKET subscript_list RBRACKET'
        p[0] = SubscriptedValue(p[1], p[3])

    def p_variable(self, p):
        '''variable : simple_variable
                    | subscripted_value'''
        p[0] = p[1]


    # 3.2 Function designators
    def p_procedure_identifier(self, p):
        'procedure_identifier : IDENTIFIER'
        p[0] = Identifier(p[1], p.lineno(1))

    def p_actual_param(self, p):
        '''actual_param : string
                        | expression
                        | array_identifier
                        | switch_identifier
                        | procedure_identifier'''
        p[0] = p[1]

    def p_letter_string(self, p):
        'letter_string : IDENTIFIER'
        if not p[1].isalpha():
            raise SyntaxError("must be letter string", self.lexer,
                              p[1], p.lineno(1), p.lexpos(1))

    def p_param_delimiter(self, p):
        '''param_delimiter : COMMA
                           | RPAREN letter_string COLON LPAREN'''
        pass


    def p_actual_param_list_1(self, p):
        'actual_param_list : actual_param'
        p[0] = [p[1]]

    def p_actual_param_list_2(self, p):
        'actual_param_list : actual_param_list param_delimiter actual_param'
        p[0] = p[1] + [p[3]]


    def p_actual_param_part_1(self, p):
        'actual_param_part : empty'
        p[0] = []

    def p_actual_param_part_2(self, p):
        'actual_param_part : LPAREN actual_param_list RPAREN'
        p[0] = p[2]


    def p_function_designator(self, p):
        'function_designator : procedure_identifier actual_param_part'
        p[0] = FunctionDesignator(p[1], p[2])


    # 3.3 Arithmetic expressions
    def p_adding_operator(self, p):
        '''adding_operator : PLUS
                           | MINUS'''
        p[0] = KeyWord(p[1], p.lineno(1))

    def p_multiplying_operator(self, p):
        '''multiplying_operator : TIMES
                                | DIVIDE
                                | DIV'''
        p[0] = KeyWord(p[1], p.lineno(1))

    def p_primary_1(self, p):
        '''primary : unsigned_number
                   | variable
                   | function_designator'''
        p[0] = p[1]

    def p_primary_2(self, p):
        'primary : LPAREN arith_expression RPAREN'
        p[0] = p[2]


    def p_factor_1(self, p):
        'factor : primary'
        p[0] = p[1]

    def p_factor_2(self, p):
        'factor : factor POWER primary'
        op = KeyWord(p[2], p.lineno(2))
        p[0] = BinaryOperation(p[1], op, p[3])


    def p_term_1(self, p):
        'term : factor'
        p[0] = p[1]

    def p_term_2(self, p):
        'term : term multiplying_operator factor'
        p[0] = BinaryOperation(p[1], p[2], p[3])


    def p_simple_arith_1(self, p):
        'simple_arith : term'
        p[0] = p[1]

    def p_simple_arith_2(self, p):
        'simple_arith : adding_operator term'
        p[0] = UnaryOperation(p[1], p[2])

    def p_simple_arith_3(self, p):
        'simple_arith : simple_arith adding_operator term'
        p[0] = BinaryOperation(p[1], p[2], p[3])


    def p_if_clause(self, p):
        'if_clause : IF bool_expression THEN'
        p[0] = (p[2], p.lineno(1), p.lineno(3))


    def p_arith_expression_1(self, p):
        'arith_expression : simple_arith'
        p[0] = p[1]

    def p_arith_expression_2(self, p):
        'arith_expression : if_clause simple_arith ELSE arith_expression'
        p[0] = IfExpression(p[1][0], p[1][1],
                            p[2], p[1][2],
                            p[4], p.lineno(3))


    # 3.4 Boolean expression
    def p_relational_operator(self, p):
        '''relational_operator : LESS
                               | NOTGREATER
                               | EQUAL
                               | NOTEQUAL
                               | NOTLESS
                               | GREATER'''
        p[0] = KeyWord(p[1], p.lineno(1))

    def p_relation(self, p):
        'relation : simple_arith relational_operator simple_arith'
        p[0] = BinaryOperation(p[1], p[2], p[3])


    def p_bool_primay_t(self, p):
        'bool_primary : TRUE'
        p[0] = True

    def p_bool_primay_f(self, p):
        'bool_primary : FALSE'
        p[0] = False

    def p_bool_primay_1(self, p):
        '''bool_primary : variable
                        | function_designator
                        | relation'''
        p[0] = p[1]

    def p_bool_primay_2(self, p):
        'bool_primary : LPAREN bool_expression RPAREN'
        p[0] = p[2]


    def p_bool_secondary_1(self, p):
        'bool_secondary : bool_primary'
        p[0] = p[1]

    def p_bool_secondary_2(self, p):
        'bool_secondary : NOT bool_primary'
        p[0] = UnaryOperation(KeyWord(p[1], p.lineno(1)), p[2])


    def p_bool_factor_1(self, p):
        'bool_factor : bool_secondary'
        p[0] = p[1]

    def p_bool_factor_2(self, p):
        'bool_factor : bool_factor AND bool_secondary'
        p[0] = BinaryOperation(p[1], KeyWord(p[2], p.lineno(2)), p[3])


    def p_bool_term_1(self, p):
        'bool_term : bool_factor'
        p[0] = p[1]

    def p_bool_term_2(self, p):
        'bool_term : bool_term OR bool_factor'
        p[0] = BinaryOperation(p[1], KeyWord(p[2], p.lineno(2)), p[3])


    def p_implication_1(self, p):
        'implication : bool_term'
        p[0] = p[1]

    def p_implication_2(self, p):
        'implication : implication IMPL bool_term'
        p[0] = BinaryOperation(p[1], KeyWord(p[2], p.lineno(2)), p[3])


    def p_simple_bool_1(self, p):
        'simple_bool : implication'
        p[0] = p[1]

    def p_simple_bool_2(self, p):
        'simple_bool : simple_bool EQUIV implication'
        p[0] = BinaryOperation(p[1], KeyWord(p[2], p.lineno(2)), p[3])


    def p_bool_expression_1(self, p):
        'bool_expression : simple_bool'
        p[0] = p[1]

    def p_bool_expression_2(self, p):
        'bool_expression : if_clause simple_bool ELSE bool_expression'
        p[0] = IfExpression(p[1][0], p[1][1],
                            p[2], p[1][2],
                            p[4], p.lineno(3))


    # 3.5 Designational expressions
    def p_label_1(self, p):
        'label : IDENTIFIER'
        p[0] = Identifier(p[1], p.lineno(1))

    def p_label_2(self, p):
        'label : UNSIGNEDINTEGER'
        p[0] = Identifier(p[1], p.lineno(1))


    def p_switch_identifier(self, p):
        'switch_identifier : IDENTIFIER'
        p[0] = Identifier(p[1], p.lineno(1))

    def p_switch_desig(self, p):
        'sw_desig : switch_identifier LBRACKET subscript_expression RBRACKET'
        p[0] = SwitchDesignator(p[1], p[3])


    def p_simple_desig_1(self, p):
        '''simple_desig : label
                        | sw_desig'''
        p[0] = p[1]

    def p_simple_desig_2(self, p):
        'simple_desig : LPAREN desig_expression RPAREN'
        p[0] = p[2]


    def p_desig_expression_1(self, p):
        'desig_expression : simple_desig'
        p[0] = p[1]

    def p_desig_expression_2(self, p):
        'desig_expression : if_clause simple_desig ELSE desig_expression'
        p[0] = IfExpression(p[1][0], p[1][1],
                            p[2], p[1][2],
                            p[4], p.lineno(3))


    # 4.2 Assignment statements
    def p_destination(self, p):
        '''destination : variable
                       | procedure_identifier'''
        p[0] = p[1]

    def p_left_part(self, p):
        'left_part : destination COLONEQUAL'
        p[0] = p[1]

    def p_left_part_list_1(self, p):
        'left_part_list : left_part'
        p[0] = [p[1]]

    def p_left_part_list(self, p):
        'left_part_list : left_part_list left_part'
        p[0] = p[1] + [p[2]]

    def p_assignment_statement(self, p):
        '''assignment_statement : left_part_list arith_expression
                                | left_part_list bool_expression'''
        p[0] = AssignmentStatement(p[1], p[2])


    # 4.3 Go to statements
    def p_goto_statement_1(self, p):
        'goto_statement : GOTO desig_expression'
        p[0] = GoToStatement(p[2], p.lineno(1))

    def p_goto_statement_2(self, p):
        'goto_statement : GO TO desig_expression'
        p[0] = GoToStatement(p[3], p.lineno(1))


    # 4.4 Dummy statements
    def p_dummy_statement(self, p):
        'dummy_statement : empty'
        p[0] = DummyStatement()


    # 4.5 Conditional statements
    def p_if_statement(self, p):
        'if_statement : if_clause unconditional_statement'
        p[0] = (p[1], p[2])     # ((if_exp, if_lineno, then_lineno), then_s)


    def p_conditional_statment_1(self, p):
        'conditional_statement : if_statement'
        (if_exp, if_lineno, then_lineno) = p[1][0]
        then_s = p[1][1]
        p[0] = IfStatement(if_exp, if_lineno, then_s, then_lineno)

    def p_conditional_statment_2(self, p):
        'conditional_statement : if_statement ELSE statement'
        (if_exp, if_lineno, then_lineno) = p[1][0]
        then_s = p[1][1]
        else_s = p[3]
        else_lineno = p.lineno(2)
        p[0] = IfStatement(if_exp, if_lineno, then_s, then_lineno,
                        else_s, else_lineno)

    def p_conditional_statment_3(self, p):
        'conditional_statement : if_clause for_statement'
        (if_exp, if_lineno, then_lineno) = p[1]
        then_s = p[2]
        p[0] = IfStatement(if_exp, if_lineno, then_s, then_lineno)

    def p_conditional_statment_4(self, p):
        'conditional_statement : label COLON conditional_statement'
        p[0] = LabelledStatement(p[1], p[3])


    # 4.6 For statements
    def p_for_list_element_1(self, p):
        'fle : arith_expression'
        p[0] = p[1]

    def p_for_list_element_2(self, p):
        'fle : arith_expression STEP arith_expression UNTIL arith_expression' 
        op = KeyWord(p[2], p.lineno(2))
        p[0] = Operation(op, p[1], p[3], p[5])

    def p_for_list_element_3(self, p):
        'fle : arith_expression WHILE bool_expression'
        op = KeyWord(p[2], p.lineno(2))
        p[0] = Operation(op, p[1], p[3])


    def p_for_list_1(self, p):
        'for_list : fle'
        p[0] = [p[1]]

    def p_for_list_2(self, p):
        'for_list : for_list COMMA fle'
        p[0] = p[1] + [p[3]]


    def p_for_clause(self, p):
        'for_clause : FOR variable COLONEQUAL for_list DO'
        p[0] = (p.lineno(1), p[2], p[4], p.lineno(5))


    def p_for_statement_1(self, p):
        'for_statement : for_clause statement'
        (for_lineno, variable, for_list, do_lineno) = p[1]
        p[0] = ForStatement(for_lineno, variable, for_list, do_lineno, p[2])

    def p_for_statement_2(self, p):
        'for_statement : label COLON for_statement'
        p[0] = LabelledStatement(p[1], p[3])


    # 4.7 Procedure statements
    def p_procedure_statement(self, p):
        'procedure_statement : procedure_identifier actual_param_part'
        p[0] = ProcedureStatement(p[1], p[2])


    # 5 Declarations
    def p_declaration(self, p):
        '''declaration : type_declaration
                       | array_declaration
                       | switch_declaration
                       | procedure_declaration'''
        p[0] = p[1]

    # 5.1 Type declarations
    def p_type_list_1(self, p):
        'type_list : simple_variable'
        p[0] = [p[1]]

    def p_type_list_2(self, p):
        'type_list : simple_variable COMMA type_list'
        p[0] = [p[1]] + p[3]


    def p_type(self, p):
        '''type : REAL
                | INTEGER
                | BOOLEAN'''
        p[0] = KeyWord(p[1], p.lineno(1))


    def p_type_declation_1(self, p):
        'type_declaration : type type_list'
        identities = p[2]
        p[0] = [SimpleVariableQuantity(id, p[1], False) for id in identities]

    def p_type_declation_2(self, p):
        'type_declaration : OWN type type_list'
        identities = p[3]
        p[0] = [SimpleVariableQuantity(id, p[2], True) for id in identities]


    # 5.2 Array declarations
    def p_lower_bound(self, p):
        'lower_bound : arith_expression'
        p[0] = p[1]

    def p_upper_bound(self, p):
        'upper_bound : arith_expression'
        p[0] = p[1]

    def p_bound_pair(self, p):
        'bound_pair : lower_bound COLON upper_bound'
        op = KeyWord(p[2], p.lineno(2))
        p[0] = BinaryOperation(p[1], op, p[3])


    def p_bound_pair_list_1(self, p):
        'bound_pair_list : bound_pair'
        p[0] = [p[1]]

    def p_bound_pair_list_2(self, p):
        'bound_pair_list : bound_pair_list COMMA bound_pair'
        p[0] = p[1] + [p[3]]


    def p_array_segment_1(self, p):
        'array_segment : array_identifier LBRACKET bound_pair_list RBRACKET'
        p[0] = [ArrayQuantity(p[1], p[3])]

    def p_array_segment_2(self, p):
        'array_segment : array_identifier COMMA array_segment'
        seg = p[3]
        a = ArrayQuantity(p[1], seg[0].bound_pair_list)
        p[0] = [a] + seg


    def p_array_list_1(self, p):
        'array_list : array_segment'
        p[0] = p[1]

    def p_array_list_2(self, p):
        'array_list : array_list COMMA array_segment'
        p[0] = p[1] + p[3]


    def p_array_declarer_1(self, p):
        'array_declarer : type ARRAY'
        p[0] = p[1]

    def p_array_declarer_2(self, p):
        'array_declarer : ARRAY'
        p[0] = KeyWord('real', p.lineno(1))


    def p_array_declation_1(self, p):
        'array_declaration : array_declarer array_list'
        for a in p[2]:
            a.is_own = False
            a.type = p[1]
        p[0] = p[2]

    def p_array_declation_2(self, p):
        'array_declaration : OWN array_declarer array_list'
        for a in p[3]:
            a.is_own = True
            a.type = p[2]
        p[0] = p[3]


    # 5.3 Switch designators
    def p_switch_list_1(self, p):
        'switch_list : desig_expression'
        p[0] = [p[1]]

    def p_switch_list_2(self, p):
        'switch_list : switch_list COMMA desig_expression'
        p[0] = p[1] + [p[3]]


    def p_switch_declaration(self, p):
        'switch_declaration : SWITCH switch_identifier COLONEQUAL switch_list'
        p[0] = [SwitchQuantity(p[2], p[4])]


    # 5.4 Procedure declarations
    def p_formal_param(self, p):
        'formal_param : IDENTIFIER'
        p[0] = Identifier(p[1], p.lineno(1))


    def p_formal_param_list_1(self, p):
        'formal_param_list : formal_param'
        p[0] = [p[1]]

    def p_formal_param_list_2(self, p):
        'formal_param_list : formal_param_list param_delimiter formal_param'
        p[0] = p[1] + [p[3]]


    def p_formal_param_part_1(self, p):
        'fparams : empty'
        p[0] = []

    def p_formal_param_part_2(self, p):
        'fparams : LPAREN formal_param_list RPAREN'
        p[0] = p[2]


    def p_identifier_list_1(self, p):
        'identifier_list : IDENTIFIER'
        p[0] = [Identifier(p[1], p.lineno(1))]

    def p_identifier_list_2(self, p):
        'identifier_list : identifier_list COMMA IDENTIFIER'
        p[0] = p[1] + [Identifier(p[3], p.lineno(3))]


    def p_value_part_1(self, p):
        'values : VALUE identifier_list semicolon'
        p[0] = p[2]

    def p_value_part_2(self, p):
        'values : empty'
        p[0] = []


    def p_specifier_1(self, p):
        'specifier : type'
        p[0] = (SimpleVariableQuantity, p[1])

    def p_specifier_2(self, p):
        'specifier : array_declarer'
        p[0] = (ArrayQuantity, p[1])

    def p_specifier_3(self, p):
        'specifier : STRING'
        p[0] = (SimpleVariableQuantity, KeyWord(p[1], p.lineno(1)))

    def p_specifier_4(self, p):
        'specifier : LABEL'
        p[0] = (LabelQuantity, None)

    def p_specifier_5(self, p):
        'specifier : SWITCH'
        p[0] = (SwitchQuantity, None)

    def p_specifier_6(self, p):
        'specifier : PROCEDURE'
        p[0] = (ProcedureQuantity, None)

    def p_specifier_7(self, p):
        'specifier : type PROCEDURE'
        p[0] = (ProcedureQuantity, p[1])


    def p_specification_part_1(self, p):
        'specs : empty'
        p[0] = []

    def p_specification_part_2(self, p):
        'specs : specifier identifier_list semicolon specs'
        p[0] = [(p[1], p[2])] + p[4]


    def p_proc_heading(self, p):
        'proc_heading : procedure_identifier fparams semicolon values specs'
        p[0] = (p[1], p[2], p[4], p[5])

    def p_proc_body(self, p):
        'proc_body : statement'     # | code
        p[0] = p[1]

    def p_prcedure_declaration_proc(self, p):
        'procedure_declaration : PROCEDURE proc_heading proc_body'
        (name, params, values, specs) = p[2]
        p[0] = [ProcedureQuantity(name, None, params, values, specs, p[3])]

    def p_prcedure_declaration_func(self, p):
        'procedure_declaration : type PROCEDURE proc_heading proc_body'
        (name, params, values, specs) = p[3]
        p[0] = [ProcedureQuantity(name, p[1], params, values, specs, p[4])]


if __name__ == '__main__':
    import pprint
    source_text = """
  begin
    comment
      Repeat reading an integer and printing the factorial of the integer
      until the integer < 0;

    integer procedure Fac(k);
      value k; integer k;
      begin
        if k = 0 then Fac := 1
                 else Fac := k * Fac(k - 1)
      end Fac;

    integer n;
  LOOP:
    read(n);
    if n < 0 then goto EXIT;
    print(n, Fac(n));
    goto LOOP;
  EXIT:
  end
    """
    syntax_tree = Parser().parse(source_text)
    pprint.pprint(syntax_tree)
