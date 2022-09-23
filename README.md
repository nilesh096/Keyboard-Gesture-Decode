# Aim
Given a dictionary of 10000 words, implement the SHARK2 algorithm to decode a user input gesture and output the best decoded word.


# Implementation
The complete implementation of this homework is with the help of the SHARK2 publication authored by Dr. Per Ola Kristensson and Dr. Shumin Zhai.
Check `server.py`

# Steps


1. Sampling

As part of sampling we first calculate the difference between the adjacent
points in x-plane and y-plane for which the function ediff1d from numpy library is
used. We also use the argument to_begin = 0 so that, adjacent difference starts from
the first index and shape mismatch error is avoided. Once we have this difference, we
then calculate the pairwise Euclidean distance and then the cumulative sum is
calculated. Since, we can have words which repeat more than once like “hh”, “mm” that lie at the
same position, we avoid this ambiguity by using centroid location. Otherwise, we get
the proportion of line segments and scale the points by doing linear interpolation
along the observed path of the which we have the coordinates. To have the 100 points
at same distance from each other, we use the linespace method of numpy library to
generate the equidistant points on the normalized line and then transform the points
from normalized plane to the cartesian coordinate plane or real plane.

2. Do Pruning

Since we want to avoid processing a large number of words which won’t
match the gesture pattern, we prune/filter out those words. For this, we calculate the
start to start and end to end distances of the template and the input gesture, if either
of those distances are greater than the threshold, we will discard that template. We
calculate the proportional matching distance given by the formula of `sqrt((sqr(x1-x0) + sqr(y1-y0)))`. The do pruning function returned the valid words stored in a list along with its probabilities. For setting the threshold, we experimented with various values
and narrowed down to 18, since with we were able to correctly predict the actual
word corresponding to the actual gesture.

3. Get_shape_scores

For this method, to calculate the shape scores, we first normalize
the pointsin scale using the function already provided i.e getScaledPoints which scales
the larger side of the bounding box to a length of 1. We the translate the pattern’s
geometric centroid to the origin in the coordinate system. The centroid is calculated
using the formula `(max(gesture_X) - min(gesture_X))/2` and normalization is done in
location. This normalization is done both for gestures and template.
Now that the normalization is done, we calculate the sum of Euclidean distances
between the gestures and the template sample points and then for each template, we
append this sum for each template (shape score) and gesture combination to a list by
dividing the sum by 100 i.e average sum of the equidistant sample points. Thus, we
will get the shape score of every valid word after pruning

4. Get_location_score

Since the shape score alone is not enough to correctly recognize
the input gesture, we also use the location channel or location score to get the
accurate recognition of the gesture to a word. We set the radius of the r of the
alphabetical key as 15. We calculate the sum of maximum of the minimum of the
differences between the gesture and the template for all valid words If the distance is
0 then it means that the entire input gesture is within the tunnel width of the
estimated gesture. Instead of using the provided helper methods, I defined my own
helper methods namely location_score_helper1 and location_score_helper_2 and
helper. Thus, we get the location scores of every user gesture and template pair.

5. get_integration_scores

In this method, we set the weight of the each score i.e location and shape. I have given 0.5 weightage to shape and 0.5 weightage to location.

6. Get_best_score

In this we select the top 7 words recommended based on the lowest integration score. No word found is returned if there were no valid words found.


# References:

1. http://pokristensson.com/pubs/KristenssonZhaiUIST2004.pdf
2. https://stackoverflow.com/questions/51512197/python-equidistant-points-along-a-line-joining-set-of-points/51515357
3. https://docs.scipy.org/doc/numpy/reference/generated/numpy.ediff1d.html
4. https://docs.scipy.org/doc/numpy/reference/generated/numpy.cumsum.html
5. https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
6. https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html
*********************************************************************************************************************

