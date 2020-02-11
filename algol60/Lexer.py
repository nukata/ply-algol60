# H20.10.06/R02.02.11 by SUZUKI Hisao

from .ply import lex

class SyntaxError (Exception):
    def __init__(self, msg, LEXER, src, lineno=None, lexpos=None):
        if lineno is None:
            lineno = LEXER.lex.lineno
        if lexpos is None:
            lexpos = LEXER.lex.lexpos
        msg += " at %d column %d: %r" % (
            lineno, lexpos - LEXER.line_head_pos, src)
        Exception.__init__(self, msg)

class Lexer:
    t_ignore = ' \t\r\f\v'
    t_ignore_comment = r'\#.*'  # Unofficial: '#' starts a line comment.

    def t_newline(self, t):
        r'\n'
        t.lexer.lineno += 1
        self.line_head_pos = t.lexpos + 1

    def t_error(self, t):
        s = t.value[0]
        raise SyntaxError("bad character", self, t.value[0])

    keywords = {
        'begin': 'BEGIN',
        'end': 'END',
        'if': 'IF',
        'then': 'THEN',
        'else': 'ELSE',
        'for': 'FOR',
        'do': 'DO',
        'go': 'GO',
        'to': 'TO',
        'goto': 'GOTO',
        'step': 'STEP',
        'until': 'UNTIL',
        'while': 'WHILE',
        'comment': 'COMMENT',
        'real': 'REAL',
        'integer': 'INTEGER',
        'Boolean': 'BOOLEAN',
        'own': 'OWN',
        'array': 'ARRAY',
        'switch': 'SWITCH',
        'procedure': 'PROCEDURE',
        'label': 'LABEL',
        'string': 'STRING',
        'value': 'VALUE',
        'true': 'TRUE',
        'false': 'FALSE',
        'not': 'NOT',
        'and': 'AND',
        'or': 'OR',
        'impl': 'IMPL',
        'equiv': 'EQUIV',
        'div': 'DIV'
        }

    tokens = [
        'POWER', 'COLONEQUAL', 'NOTEQUAL', 'NOTLESS', 'NOTGREATER',
        'EQUAL', 'LESS',  'GREATER', 
        'PLUS', 'MINUS', 'TIMES', 'DIVIDE',
        'COMMA', 'COLON', 'SEMICOLON',
        'LPAREN', 'RPAREN', 'LBRACKET', 'RBRACKET',
        'IDENTIFIER',
        'CLOSEDSTRING',
        'UNSIGNEDREAL',
        'UNSIGNEDINTEGER'
        ] + list(keywords.values())

    t_POWER = r'\*\*'
    t_COLONEQUAL = ':='
    t_NOTEQUAL = '/='
    t_NOTLESS = '>='
    t_NOTGREATER = '<='
    t_EQUAL = '='
    t_LESS = '<'
    t_GREATER = '>'
    t_PLUS = r'\+'
    t_MINUS = '-'
    t_TIMES = r'\*'
    t_DIVIDE = '/'
    t_COMMA = ','
    t_COLON = ':'
    t_SEMICOLON = ';'
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_LBRACKET = r'\['
    t_RBRACKET = r'\]'

    states = [
        ('com', 'exclusive'),
        ('end', 'exclusive'),
        ('str', 'exclusive')
        ]

    def t_IDENTIFIER(self, t):
        r'_?[A-Za-z][0-9A-Za-z]*' # Unofficial: inentifiers may begin with _.
        t.type = self.keywords.get(t.value, 'IDENTIFIER')
        if t.type == 'END':
            t.lexer.push_state('end')
        elif t.type == 'COMMENT':
            t.lexer.push_state('com')
        return t

    # "end" <any sequence of zero or more characters not containing 
    # "end" or ";" or "else">
    t_end_ignore = t_ignore
    t_end_newline = t_newline
    def t_end_error(self, t): t.lexer.skip(1)

    def t_end_END(self, t):
        r'end'
        t.type = 'END'
        return t

    def t_end_ELSE(self, t):
        r'else'
        t.lexer.pop_state()
        t.type = 'ELSE'
        return t
        
    def t_end_SEMICOLON(self, t):
        r';'
        t.lexer.pop_state()
        t.type = 'SEMICOLON'
        return t

    # "comment" <any sequence of zero or more characters not containing
    # ";"> ";"
    t_com_ignore = t_ignore
    t_com_newline = t_newline
    def t_com_error(self, t): t.lexer.skip(1)

    def t_com_SEMICOLON(self, t):
        r';'
        t.lexer.pop_state()


    # <closed string> ::= "`" <open string> "'"
    def t_begin_str(self, t):
        r"`"
        self.str_start = t.lexer.lexpos
        self.str_level = 1
        t.lexer.push_state('str')

    t_str_ignore = t_ignore
    t_str_newline = t_newline
    def t_str_error(self, t): t.lexer.skip(1)
    
    def t_str_begin(self, t):
        r"`"
        self.str_level += 1

    def t_str_end(self, t):
        r"'"
        self.str_level -= 1
        if self.str_level == 0:
            t.lexer.pop_state()
            t.type = 'CLOSEDSTRING'
            t.value = t.lexer.lexdata[self.str_start: t.lexer.lexpos - 1]
            return t

    def t_real1(self, t):
        r'(\d(_?\d)*)?\.\d(_?\d)*(e(\+|-)?\d(_?\d)*)?'
        t.type = 'UNSIGNEDREAL'
        t.value = ''.join(t.value.split('_'))
        return t

    def t_real2(self, t):
        r'\d(_?\d)*e(\+|-)?\d(_?\d)*'
        t.type = 'UNSIGNEDREAL'
        t.value = ''.join(t.value.split('_'))
        return t

    def t_UNSIGNEDINTEGER(self, t):
        r'\d(_?\d)*'
        t.value = ''.join(t.value.split('_'))
        return t


    def token(self):
        return self.lex.token()

    def input(self, text):
        self.lex = lex.lex(object=self)
        self.line_head_pos = 0
        self.lex.input(text)

    def test(self, text):
        try:
            self.input(text)
            while True:
                tok = self.token()
                if not tok:
                    break
                print(tok)
        except SyntaxError as ex:
            print(ex)

if __name__ == '__main__':
    Lexer().test("""
    foo(a, b) bar:(c); 
    value real integer Boolean boolean
    begin comment a b c && d & jjj;
    print(`this is' `a `string'');
    for a := 10e3, .567, 2.0
    = /= < > <= >= [] a*b**c
    end 1 end 2; @
    """)
