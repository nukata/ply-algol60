# Algol 60 interpreter in Python with PLY

This is an Algol 60 interpreter written in Python with
Beazley's [PLY](https://github.com/dabeaz/ply) (Python Lex-Yacc).

I wrote it in Python 2.3/3.0 with PLY 3.0 in 2008 (H20) - 2009 (H21)
and presented it under the MIT License at
<http://www.oki-osk.jp/esc/ply-algol/> (broken link)
until the spring of 2017 (H29).
Now I have slightly modified it to match Python 2.7/3.7 and PLY 3.11.

## The language

This interpreter implements the specification written in
[Modified Report on the Algorithmic Language Algol 60](https://www.masswerk.at/algol60/modified_report.htm)
with the following _hardware representation_:

* Keywords are spelled as they are in lower case except for `Boolean`.

* Spaces are not allowed within the spelling of keywords and identifiers;
  `goto`, `go` and `to` are keywords.

* ≠, ≦, ¬ are written as `/=`, `<=`, `not` respectively.

* ↑, ÷, ≡, ⊃, ∨, ∧ are written as `**`, `div`, `equiv`, `impl`, `or`, `and`
  respectively.

* <sub>10</sub> is written as `e`.
  Note that <sub>10</sub>-4 is written as `1e-4` (not `e-4`).


## Corrections

I found two mistakes in the report and corrected them in the interpreter.

In 4.5 "Conditional statements", 4.5.1. "Syntax" of the report:
```
<conditional statement> ::= <if statement> |
        <if statement> else <statement> |
        <if clause> <for statement> |
        <block>: <conditional statement>
```

In the interpreter it is corrected as follows:
```
<conditional statement> ::= <if statement> |
        <if statement> else <statement> |
        <if clause> <for statement> |
        <label>: <conditional statement>
```

See [algol60/Parser.py](algol60/Parser.py#L517).


In Appendix 2 "The environmental block", the procedure `ininteger`
includes the following code:

```
      if k > 10 then
         begin
            comment sign found, d indicates digit not found yet, b 
            indicates whether + or -, m is value so far;
            d := b := true;
            m := k - 1
         end;
```

In the interpreter it is corrected as follows:

```
      if k > 10 then
         begin
            comment sign found, d indicates digit found, b
               indicates the sign, m is value so far;
            d := false;
            b := k /= 11;
            m := 0
         end
      else
         begin
            d := b := true;
            m := k - 1
         end;
```

See [algol60/Prelude.py](algol60/Prelude.py#L174-L186).


## How to run

The interpreter has no third-party dependencies
(PLY is included in `algol60` module).
Just use `python`, `python3`, `pypy` or `pypy3`.

```
$ python -m algol60 demo/factorial.txt
n := 1
1 1
n := 10
10 3628800
n := 20
20 2432902008176640000
n := 30
30 265252859812191058636308480000000
n := 40
40 815915283247897734345611269596115894272000000000
n := 50
50 30414093201713378043612608166064768844377641568960512000000000000
n := -1
$ 
```

```
$ pypy3 -m algol60 demo/tak.txt
7
$ 
```

Or use a small shell script `ply-algol60`.

```
$ cat ply-algol60
#!/bin/sh -
exec python -m algol60 "$@"
$ ./ply-algol60 demo/absmax.txt
ans := 21.20000000000000  at 1 2
$ 
```
