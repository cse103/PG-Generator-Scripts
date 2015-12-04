import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
from numpy.random import shuffle
from scipy.stats import norm, uniform, expon

num_question = 5

header = """
DOCUMENT();        # This should be the first executable line in the problem.

loadMacros(
  "PGstandard.pl",
  "PGML.pl",
  "MathObjects.pl",
  "PGcourse.pl",
);

Context("Numeric");

Context()->flags->set(tolerance=>0.01);

TEXT(beginproblem);
$showPartialCorrectAnswers = 1;

$rand_question = random(0,%d,1);
######################################################################
""" % (num_question-1)

footer = """
ENDDOCUMENT();        # This should be the last executable line in the problem.
"""

pg = open('cdf_random.pg', 'w')
pg.write(header)


draw_components = False


low_lim = -5
high_lim = 5
dx = .001
X = np.arange(low_lim,high_lim,dx)

configurations = [[2,2,0,0]] # 2 points masses, 2 uniforms, no exponential or gaussian

for fig_idx in range(num_question):

	point_num, uniform_num, use_normal, use_exp = configurations[randint(0,len(configurations))]

	# point_num = randint(1,3)
	# point_num = 1
	# uniform_num = randint(1,3)
	# uniform_num = 1
	# use_normal = 1
	# use_exp = 0

	# uniform distribution components
	upper_bound = [2, 5]
	lower_bound = [-5, -2]
	if uniform_num > 0:
		uniform_res = 0.5
		low = np.zeros(uniform_num)
		high = np.zeros(uniform_num)
		uniform_cdf = np.zeros((uniform_num, len(X)))
		uniform_pdf = np.zeros((uniform_num, len(X)))
		rv_uniform = [None]*uniform_num
		for uniform_idx in range(uniform_num):
			while True:
				low[uniform_idx], high[uniform_idx] = np.sort(np.random.randint((low_lim)/uniform_res, (high_lim)/uniform_res, 2))*uniform_res
				if low[uniform_idx] != high[uniform_idx] and high[uniform_idx] < upper_bound[uniform_idx] and low[uniform_idx] > lower_bound[uniform_idx]:
					if uniform_idx == 0 and high[0] > -2 and low[0] < -2:
						break
					elif uniform_idx == 1 and low[1] < high[0] and high[1] > high[0]:
						break
			print low[uniform_idx], high[uniform_idx]
			rv_uniform[uniform_idx] = uniform(low[uniform_idx], high[uniform_idx]-low[uniform_idx])
			uniform_pdf[uniform_idx] = rv_uniform[uniform_idx].pdf(X)
			uniform_cdf[uniform_idx] = rv_uniform[uniform_idx].cdf(X)

	# point mass components
	if point_num > 0:
		point_res = 0.5
		while True: 
			locations = randint((low_lim+1) /point_res, (high_lim-1)/point_res, point_num)*point_res
			if len(set(locations))!=len(locations):
				continue
			if uniform_num>0:
				if np.equal.outer(locations,low).any() and np.equal.outer(locations,high).any():
					continue
			if use_normal:
				if (locations==mean).any():
					continue
			break

		point_cdf = np.zeros((point_num, len(X)))
		for point_idx in range(point_num):
			point_cdf[point_idx, int((locations[point_idx]-low_lim)/dx):] = 1.


	# component weights
	while True:
		# ws = np.random.dirichlet(np.ones(1+1+point_num), 1)[0]
		# w_normal = ws[0]
		# w_uniform = ws[1]
		# w_points = ws[2:]
		num_component = uniform_num + point_num + use_normal + use_exp
		s = list(sorted(randint(1,19,num_component-1)))
		#l = [i for i in range(2,19,2)]
		#shuffle(l)
		#s = list(sorted(l[:num_component-1]))
		split = [0] + s + [20]
		component_weights = np.diff(split)*0.05
		if component_weights.min() >= 1./(num_component+2):
			break
	w_uniform = component_weights[:uniform_num]
	w_points = component_weights[uniform_num:uniform_num+point_num]
	w_normal = component_weights[uniform_num+point_num] if use_normal else 0
	w_exp = component_weights[uniform_num+point_num+use_normal] if use_exp else 0
	print split
	print w_uniform

	# plot densities, compute and plot mixture cdf
	all_cdf = np.zeros(len(X))

	component_list = ''

	if uniform_num>0:
		if draw_components:
			plt.plot(X, np.dot(np.diag(w_uniform), uniform_pdf).T, 'b')	
		for uniform_idx in range(uniform_num):
			component_list += "\n- Uniform component on the interval ([_______]{%s},[_____]{%s}). Its component weight is [_____]{%s}"\
						 % (low[uniform_idx],high[uniform_idx],np.around(w_uniform[uniform_idx], decimals=2))
		all_cdf += np.dot(uniform_cdf.T, w_uniform)

	if point_num>0:
		if draw_components:
			markerline, stemlines, baseline = plt.stem(locations, w_points, '-.')
			plt.setp(markerline, 'markerfacecolor', 'g')
			plt.setp(stemlines, 'color', 'g')
		for point_idx in range(point_num):
			component_list += "\n- Point mass on [_______]{%s}. Its component weight is [_____]{%s}"\
						% (locations[point_idx], np.around(w_points[point_idx], decimals=2))
		all_cdf += np.dot(point_cdf.T, w_points)
	
	plt.plot(X, all_cdf, 'k')
	
	# plot axis ticks
	plt.tick_params(axis='both', labelsize=10)
	locs = np.array(list(set(
							([mean] if w_normal>0 else []) + 
							([0] if w_exp>0 else []) +
							((list(low) + list(high)) if uniform_num>0 else []) + 
							(list(locations) if point_num>0 else [])+
							[low_lim, high_lim]
							)))
	plt.xticks(locs, list(locs))

	indices = [int((i-low_lim)/dx) for i in locs if i not in [low_lim, high_lim]] + [-1]
	ylocs = [all_cdf[i] for i in indices] + ([all_cdf[int((i-low_lim)/dx-1)] for i in locations] if point_num>0 else [])
	plt.yticks(ylocs, [np.around(i, decimals=2) for i in ylocs])
	
	plt.grid()
	# plt.show()
	png_name = 'mixture_cdf_%d.png'%fig_idx
	plt.savefig(png_name, bbox_inches='tight')
	plt.clf()

	question_body = """
if ($rand_question == %d) {
BEGIN_PGML
Below is the CDF of a mixture distribution with point mass and uniform components.

All parameters of component distributions are small multiples of 0.5.

Component weights take on multiples of 0.05 and they need to sum to one.

END_PGML
BEGIN_TEXT
$BR
\{ image("%s", width=>400, height=>320) \}
END_TEXT
BEGIN_PGML

Identify the component distributions:
%s
END_PGML
}
""" % (fig_idx, png_name, component_list)

	pg.write(question_body)

pg.write(footer)
pg.close()
