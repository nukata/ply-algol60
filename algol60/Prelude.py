# H20.12.03/R02.02.11 by SUZUKI Hisao

import math, sys

def get_file(channel, mode):
    f = channels.get(channel)
    if f is None:
        channels[channel] = f = os.fdopen(channel, mode)
    return f

channels = {0: sys.stdin, 1: sys.stdout, 2: sys.stderr}

def inchar(channel, s):
    rf = get_file(channel, "r")
    ch = rf.read(1)
    r = s.find(ch)
    return r + 1

def outchar(interp):
    def _outchar(channel, s, i):
        rf = get_file(channel, "w")
        ch = s[i - 1]
        rf.write(ch)
        if channel == 1:
            interp.in_new_line = (ch == '\n')
    return _outchar

def maxint():
    try:
        return sys.maxint
    except AttributeError:
        return sys.maxsize      # for Python 3.*

def entier(e):
    return int(math.floor(e))

# The following is based on "Appendix 2 The environmental block" in
# "Modified Report on the Algorithmic Language ALGOL 60" (1976)
# with a correction to "ininteger" procedure.

PRELUDE = """
begin
   comment Simple functions;
   real procedure abs(E);
      value E; real E;
      abs := if E >= 0.0 then E else -E;

   integer procedure iabs(E);
      value E; integer E;
      iabs := if E >= 0 then E else -E;

   integer procedure sign(E);
      value E; integer E;
      sign := if E > 0.0 then 1
         else if E < 0.0 then -1 else 0;

   integer procedure entier(E);
      value E; real E;
      comment entier := largest integer not greater than E,
         i.e. E - 1 < entier <= E;
      entier := _nativecall(`Prelude.entier', E);


   comment Mathematical functions;
   real procedure sqrt(E);
      value E; real E;
      if E < 0.0 then
         fault(`negative sqrt', E)
      else
         sqrt := E**0.5;

   real procedure sin(E);
      value E; real E;
      comment sin := sine of E radians;
      sin := _nativecall(`Prelude.math.sin', E);

   real procedure cos(E);
      value E; real E;
      cos := _nativecall(`Prelude.math.cos', E);

   real procedure arctan(E);
      value E; real E;
      arctan := _nativecall(`Prelude.math.atan', E);

   real procedure ln(E);
      value E; real E;
      comment ln := natural logarithm of E;
      if E <= 0.0 then
         fault(`ln not positive', E)
      else
         ln := _nativecall(`Prelude.math.log', E);

   real procedure exp(E);
      value E; real E;
      comment exp := exponential function of E;
      if E > ln(maxreal) then
         fault(`overflow on exp', E)
      else
         exp := _nativecall(`Prelude.math.exp', E);


   comment Terminating procedures;
   procedure stop;
       go to _stoplabel;

   procedure fault(str, r);
      value r; string str; real r;
   begin
      print(`fault', `', str, `', r);
      stop
   end fault;

 
   comment Input/output procedures;
   procedure inchar(channel, str, int);
      value channel;
      integer channel, int; string str;
      comment Set int to value corresponding to the first position in
         str of current character on channel.  Set int to zero if 
         character not in str.  Move channel pointer to next character;
      int := _nativecall(`Prelude.inchar', channel, str);

   procedure outchar(channel, str, int);
      value channel, int;
      integer channel, int; string str;
      comment Pass to channel the character in str, corresponding to
         the value of int;
      if int < 1 or int > length(str) then
         fault(`character not in string', int)
      else
         _nativecall(`Prelude.outchar(self)', channel, str, int);

   integer procedure length(str);
      string str;
      comment length := number of characters in the string;
      length := _nativecall(`len', str);

   procedure outstring(channel, str);
      value channel;
      integer channel; string str;
   begin
      integer m, n;
      n := length(str);
      for m := 1 step 1 until n do
         outchar(channel, str, m)
   end outstring;

   procedure outterminator(channel);
      value channel; integer channel;
      comment outputs a terminator for use after a number;
      outchar(channel, ` ', 1);


   procedure ininteger(channel, int);
      value channel; integer channel, int;
      comment int takes the value of an integer;
   begin
      integer k, m;
      Boolean b, d;
      integer procedure ins;
      begin
         integer n;
         comment read one character, converting newlines to spaces;
         inchar(channel, `0123456789-+ ;`NL'', n);
         ins := if n = 15 then 13 else n
      end ins;

      comment pass over initial spaces or newlines;
      for k := ins while k = 13 do
         ;
      comment fault anything except sign or digit;
      if k = 0 or k > 13 then
         fault(`invalid character', k);
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
      for k := ins while k > 0 and k < 11 do
         begin
            comment deal with further digits;
            m := 10 * m + k - 1;
            d := true
         end k loop;
      comment fault if not digit has been found, or the terminator
         was invalid;
      if d impl k < 13 then
         fault(`invalid character', k);
      int := if b then m else -m
   end ininteger;

   procedure outinteger(channel, int);
      value channel, int;
      integer channel, int;
      comment Passes to channel the characters representing the value
         of int, followed by a terminator;
   begin
      procedure digits(int);
         value int; integer int;
      begin
         integer j;
         comment use recursion to evaluate digits from right to left,
            but print them from left to right;
         j := int div 10;
         int := int - 10 * j;
         if j /= 0 then
            digits(j);
         outchar(channel, `0123456789', int + 1)
      end digits;

      if int < 0 then 
         begin
            outchar(channel, `-', 1);
            int := -int
         end;
      digits(int); outterminator(channel)
   end outinteger;


   procedure inreal(channel, re);
      value channel;
      integer channel; real re;
   begin
      integer j, k, m;
      real r, s;
      Boolean b, d;
      integer procedure ins;
      begin
         integer n;
         comment read one character, converting newlines to spaces;
         inchar(channel, `0123456789-+.e ;`NL'', n);
         ins := if n = 17 then 15 else n
      end ins;

      comment pass over initial spaces or newlines;
      for k := ins while k = 15 do
         ;
      comment fault anything except sign, digit, point or ten;
      if k = 0 or 15 < k then
         fault(`invalid character', k);
      b := k /= 11;
      d := true;
      m := 1;
      j := if k < 11 then 2 else iabs(k + k - 23);
      r := if k < 11 then k - 1 else 0.0;
      if k /= 14 then
         begin
            comment ten not found, Continue until ten or terminator found;
            for k := ins while k < 14 do
               begin
                  comment fault for non-numerical character, sign or
                     second point;
                  if k = 0 or k = 11 or k = 12 or k = 13 and j > 2 then
                     fault(`invalid character', k);
                  comment deal with digit unless it cannot affect value;
                  if d then
                     begin
                        if k = 13 then
                           begin comment point found;
                              j := 3
                           end
                        else
                           begin
                              if j < 3 then
                                 begin comment deal with digit before point;
                                    r := 10.0 * r + k - 1
                                 end
                              else
                                 begin comment deal with digit after point;
                                    s := 10.0 ** (-m);
                                    m := m + 1;
                                    r := r + s * (k - 1);
                                    comment if r = r + s to machine accuracy,
                                       further digits cannot affect value;
                                    d := r /= r + s
                                 end;
                              if j = 1 or j = 3 then j := j + 1
                           end
                     end if d
               end k loop;
            comment fault if no digit has been found;
            if j = 1 and k /= 14 or j = 3 then
               fault(`invalid character', k);
         end;         
      if k = 14 then
         begin comment deal with exponent part;
            ininteger(channel, m);
            r := (if j = 1 or j = 5 then 1.0 else r) * 10.0 ** m
         end;
      re := if b then r else -r
   end inreal;

   procedure outreal(channel, re);
      value channel, re;
      integer channel; real re;
      comment Passes to channel the characters representing the value
         of re, followed by a terminator;
   begin  
      integer n;
      comment n gives number of digits to print;
      n := entier(1.0 - ln(epsilon) / ln(10.0));
      if re < 0.0 then
         begin
            outchar(channel, `-', 1);
            re := - re
         end;
      if re < minreal then
         outstring(channel, 0.0)
      else
         begin
            integer j, k, m, p;
            Boolean float, nines;
            comment m will hold number of places point must be moved to
               standardise value of re to have one digit before point;
            m := 0;
            comment standardise value of re;
            for m := m + 1 while re >= 10.0 do
               re := re / 10.0;
            for m := m - 1 while re < 1.0 do
               re := 10.0 * re;
            if re >= 10.0 then
               begin comment this can occur only by rounding error, but
                     is a necessary safeguard;
                  re := 1.0;
                  m := m + 1
               end;
            if m >= n or m < -2 then
               begin comment printing to be in exponent form;
                  float := true;
                  p := 1
               end
            else
               begin comment printing to be in non-exponent form;
                  float := false;
                  comment if p = 0 point will not be printed.  Otherwise
                     point will be after p didigts;
                  p := if m = n - 1 or m < 0 then 0 else m + 1;
                  if m < 0 then
                     begin
                        outstring(channel, `0.');
                        if m = -2 then
                           outchar(channel, `0', 1)
                     end
               end;
            nines := false;
            for j := 1 step 1 until n do
               begin comment if nines is true, all remaining digits must
                     be 9.  This can occur only by rounding error, but
                     is a necessary safeguard;
                  if nines then
                     k := 9
                  else
                     begin comment find digit to print;
                        k := entier(re);
                        if k > 9 then
                           begin
                              k := 9;
                              nines := true
                           end
                        else
                           begin comment move next digit to before point;
                              re := 10.0 * (re - k)
                           end
                     end;
                  outchar(channel, `0123456789', k + 1);
                  if j = p then
                     outchar(channel, `.', 1)
               end j loop;
            if float then
               begin comment print exponent part.  outinteger includes a
                     call of outterminator;
                  outchar(channel, `e', 1);
                  outinteger(channel, m)
               end
            else
               outterminator(channel)
         end
   end outreal;


   comment Environmental enquiries;

   real procedure maxreal;
      maxreal := 1.79769313486e+308;

   real procedure minreal;
      minreal := 5e-324;

   integer procedure maxint;
      maxint := _nativecall(`Prelude.maxint');

   real procedure epsilon;
      epsilon := 2e-16;

    ;
_stoplabel:
end
"""
