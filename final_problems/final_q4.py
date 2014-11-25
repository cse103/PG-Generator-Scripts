import os, sys
import itertools
import random
import numpy as np
from fractions import Fraction

def print_table(a, blank_id):
	a = np.vectorize(lambda x: x.__str__())(a)
	q = a.ravel()[:-1]
	q[blank_id] = ''
	p =	'''
BEGIN_TEXT
\{ begintable(5) \}
\{ row( "", " X=1  ", "  X=2  ",  " X=3 ", "Marginal over Y") \}
\{ row( "Y=1", " %s", "%s",  "%s",  "%s" ) \}
\{ row( "Y=2", " %s ",  "%s", "%s",  "%s" ) \}
\{ row( "Y=3", "%s", " %s", " %s ", " %s " ) \}
\{ row( "Marginal over X", " %s ", " %s ", " %s ", "*" ) \}
\{ endtable() \}
END_TEXT
''' % tuple(q)
	return p

header = '''
DOCUMENT();

loadMacros(
   "PGML.pl",
   "PGstandard.pl",     # Standard macros for PG language
   "MathObjects.pl",
   "contextArbitraryString.pl"
);

TEXT(beginproblem());
$showPartialCorrectAnswers = 1;

$seed = random(0,5,1);

BEGIN_PGML
For the following joint probability matrices, fill in the missing values and answer the questions. 
END_PGML
'''

footer = '''
ENDDOCUMENT();
'''

def write_body(perm, a1, a2, a3):
	Aperm = [[a1,a2,a3][p] for p in perm]
	tables = [print_table(a, np.random.choice(range(15), size=3, replace=False)) for a in Aperm]
	ans = [[('yes','yes'),('no','no'),('yes','no')][p] for p in perm]
	body = ''
	for a, t, (ans1, ans2) in zip(Aperm, tables, ans):
		while True:
			x1, y1, x2, y2 = np.random.randint(1,4,4)
			if x1 != x2 and y1 != y2 and a[y1-1,x1-1] != 0 and a[y2-1,x2-1] != 0:
				break

		cond_q = ["- [`P(X=%d|Y=%d) = `] [__________]{Compute(\"%s\")}" % (x1, y1, a[y1-1,x1-1]/a[y1-1,3]),
		"- [`P(Y=%d|X=%d) = `] [__________]{Compute(\"%s\")}" % (y2, x2, a[y2-1,x2-1]/a[3,x2-1])][random.randint(0,1)]

		body += '''
%s
Context("ArbitraryString");
BEGIN_PGML
- Are [`X`] and [`Y`] dependent? [__________]{"%s"}
- Are [`X`] and [`Y`] correlated? [__________]{"%s"}
END_PGML
Context("Numeric");
BEGIN_PGML
%s
---
END_PGML
''' % (t, ans1, ans2, cond_q)

	return body

def generate_a1():
	while True:
		a1 = np.zeros((4,4), dtype=np.object)
		a1[:3, :3] = np.random.randint(0,3,size=(3,3))
		if (a1[0] == a1[1]).all() and (a1[0] == a1[2]).all():
			continue
		s1 = a1.sum()
		a1 = np.vectorize( lambda x: Fraction(x, s1))(a1)
		a1[3,:3] = np.sum(a1[:3,:3], axis=0)
		a1[:3,3] = np.sum(a1[:3,:3], axis=1)
		break
	return a1

def generate_a2():
	a2 = np.zeros((4,4), dtype=np.object)

	mx = np.random.randint(1,3,size=(3,))
	sx = mx.sum()
	mx = np.vectorize(lambda x: Fraction(x, sx))(mx)
	a2[:3,3] = mx

	my = np.random.randint(1,3,size=(3,))
	sy = my.sum()
	my = np.vectorize(lambda x: Fraction(x, sy))(my)
	a2[3,:3] = my

	a2[:3,:3] = np.outer(mx, my)
	return a2

def generate_a3():
	a3 = np.zeros((4,4), dtype=np.object)
	a3[:3,:3] = np.array([[1./8,0,1./8],[0,1./2,0],[1./8,0, 1./8]])
	a3[3,:3] = np.sum(a3[:3,:3], axis=0)
	a3[:3,3] = np.sum(a3[:3,:3], axis=1)
	a3 = np.vectorize(Fraction)(a3)
	return a3

if __name__ == '__main__':
	all_perms = list(itertools.permutations(range(3),3))

	num_version = 6

	prob = header
	for i in range(num_version):
		perm = all_perms[random.randint(0, len(all_perms)-1)]
		a1 = generate_a1()
		a2 = generate_a2()
		a3 = generate_a3()
		body = write_body(perm, a1, a2, a3)
		prob += '\nif ($seed==%d) {'%i + body + '}\n'
	prob += footer

	f = open('final_q4.pg', 'w')
	f.write(prob)
	f.close()
	# f2 = open('q4.ans'%i, 'w')
	# print << f2, [[a1,a2,a3][p] for p in perm]
	# f2.close()
