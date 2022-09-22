# Aim
Given a dictionary of 10000 words, implement the SHARK2 algorithm to decode a user input gesture and output the best decoded word.


# Implementation
The complete implementation of this homework is with the help of the SHARK2 publication authored by Dr. Per Ola Kristensson and Dr. Shumin Zhai.
Check `server.py`

# Steps

1. Sampling
SHARK2 performs comparations between the user input pattern and standard templates of word. But
for this to happen, we must first make them comparable. To do that, irrespective of the length of the gesture,
we must first sample the gesture to 100(in our case) points along the pattern.
To do this, first I am calculating the Euclidean distance and storing them in a variable so that I can later 
divide this distance into equidistant points. This is achieved by using the combination of Numpy's cumsum,
sqrt and ediff1d functions. The ediff1d computes the differences between consecutive elements of an array,
sqrt returns the non-negative square-root of an array, element-wise, while cumsum returns the cumulative sum of
the elements along a given axis. After this is done, I am carrying out the process of interpolation, by using
Scipy's interp1d function, which basically, interpolates a 1-D function. Finally, I am divide the distance into
100 equidistant points, by Numpy's linspace function that returns evenly spaced numbers over a specified 
interval.This completes the process of sampling.

2. Pruning
Pruning refers to the process of computing the start-to-start and end-to-end distances between a 
template and the unknown gesture entered by the user. Once I do that I have a predetermined threshold. Words 
satisfying this threshold will be returned as valid words. I am not performing any normalization in this step.
Normalization is performed in the next step.

3. Shape Score
Before we generate the shape score, we need to normalize both the points of the valid words 
returned by the pruning step, as well as the template points. To do this, I have written a custom helper
function called, get_scaled_points. This function takes the  x-axis sample points, y-axis sample points and
the parameter L. It then carries out the task of scaling both the unknown gesture and the template points, 
by a scaling factor computed by the formula:
s=L/max(H,W)
The denominator is basically the largest side of the bounding box of the gesture.
Once we multiply the scaling factor to all the points, the next step is to move the centroid of all the points 
to the origin,which is done by the get_scaled_points, as well. Finally, we compute the Euclidean norm of the
points returned by get_scaled_points function and return the shape scores.

4. Location Score
This step is the most time-consuming step, as we do 100*100*number_of_templates calculation
here. For this step, I created three helper functions, namely, get_small_d, get_big_d and get_delta, according 
to the algorithm given in the publication. Also, I computed the alpha array in this step, which along with the
delta value, is used to compute the final location score.

5. Integration Score
Final step is to compute the integration score, which is given by the formula:
Integration Score = shape coefficient * shape score + location coefficient * location score , where 
shape coefficient + location coefficient=1.
In my case, both shape coefficient and location coefficient share the same value, that is, 0.5.

Output
The final output is the word with the lowest integration score. If there is a tie, we return all words,
sharing that lowest integration score.


# References:

1. http://pokristensson.com/pubs/KristenssonZhaiUIST2004.pdf
2. https://stackoverflow.com/questions/51512197/python-equidistant-points-along-a-line-joining-set-of-points/51515357
3. https://docs.scipy.org/doc/numpy/reference/generated/numpy.ediff1d.html
4. https://docs.scipy.org/doc/numpy/reference/generated/numpy.cumsum.html
5. https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
6. https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html
*********************************************************************************************************************

