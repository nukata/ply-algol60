# H20.12.08/R02.02.11 by SUZUKI Hisao

from __future__ import print_function
import sys
from .Parser import Parser, SyntaxError, SemanticsError
from .Interpreter import Interpreter
from . import __version__
from . import ply

if len(sys.argv) != 2:
    print("Algol60 interpreter (version %s) in Python %d.%d.%d with PLY %s" %
          ((__version__,) + sys.version_info[:3] + (ply.__version__,)),
          file=sys.stderr)
    print("   python -m algol60 source_file_name", file=sys.stderr)
    print("   python -m algol60 - < source_file_name", file=sys.stderr)
    sys.exit(1)

parser = Parser(debug=False)
interp = Interpreter(parser)

if sys.argv[1] == "-":
    source_text = sys.stdin.read()
else:
    with open(sys.argv[1]) as rf:
        source_text = rf.read()

try:
    interp.run(source_text)
except (SyntaxError, SemanticsError) as ex:
    print("%s: %s" % (ex.__class__.__name__, ex), file=sys.stderr)
    sys.exit(1)
