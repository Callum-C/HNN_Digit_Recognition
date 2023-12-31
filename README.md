# HNN_Digit_Recognition
Utilising Hopfield Neural Networks to recognise digits 0 - 9.

A network is initialised and trained on a subset of the 10 available digits.

The network excels at recognising digits it has learned so long as the number it has "memorised" is no more than 2.
![Recalling_4](https://github.com/Callum-C/HNN_Digit_Recognition/assets/60474698/86fa4a74-08f4-4561-a910-f5e9d7ca87eb)

The network is capable of recalling one of two memories even if the pattern is severely noisy or corrupted.

Due to the extreme similarity between all the digits, spurious minima occur when the network has learned 3 or more digits.

This is even the case when the 3 most dis-similar digits are learned, despite the network having a theorised capacity of 375 patterns / memories.


## Dataset
Typically, this application is tested on the MNIST dataset of digits.
However, in this test, I used a custom made dataset of 10 images, one for each digit.
There are three different sizes of images; these are 28x28, 50x50 and rather ambitiously, 280x280 where the size is given in pixels.
In case of the 28x28, pixel art was used to create blocky, 8-bit styled digits.
For the 50x50 and 280x280, digits were created using photoshop by simply typing the digit, maximising the size to fit the canvas, and utilising a white background with a black font.
Most experimentation occured on the 50x50 dataset with the 280x280 set of images requiring too much memory to create the weight matrix.
Images are normalised to pixel values of 1 and -1, -1 representing black pixels, and 1 representing white pixels.

## Hamming Distance
Hamming distance is used to compare the similarity between all the digits.
Hamming distance compares two strings of bits, and counts in how many elements, the strings differ.
The more elements that differ, the larger the distance between them.
As the network naturally excels at learning pairs of digits, Hamming distance is best applied where spurious minima occur.
Using Hamming distance, and comparing all possible combinations of digits, we can find an optimal group of digits to learn or memorise.
When looking for the three best digits to learn, we are looking for the combination that has the biggest distance between eachother.
In this dataset's case, the optimal group are 0, 4 and 7.

## Training
Hebbian and Storkey training methods are both implemented and tested.
The Storkey method saw marginal improvement, noticeably so recalling the 0 and 7 on the optimal group.
However, the Storkey method struggles recalling the 4 and falls into a spurious minima, recreating a digit seemingly comrpomised of all three memorised patterns.
The Hebbian learning method excels at memorising one or two digits, however, falls into spurious minima when recalling any digit from the optimal group.

## Potential
It's theorised that the number of patterns a HNN (Hopfield Neural Network) can successfully remember, is approximately 0.15 * N.
Where N is the number of neurons in the network. 
When working with the 50x50 dataset, the image is represented by 2500 pixels, therefore requiring 2500 neurons.
With this many neurons, the HNN should be able to successfully memorise 375 patterns or different memories.
Thus at 2, potentially 3, digits, the HNN falls drastically short of expected performance at recognising digits.
This is due to the extreme similarity between the different digits, with the worst case being the digits 0, 6, and 8. 
This similarity comes from how many pixels are used to represent each of the digits. 

## Improvements
### 1 - Neuron Optimisation
Due to the vertical nature of digits, a large amount of neurons on the left and the right of the image, are unused.
Some neurons remaining constantly "on" for all digit representations and others remaining constantly "off".
Therefore, the neurons could be better optimised by filling out the canvas more.

### 2 - Colour
In a black and white representation of the image, each pixel is only represented by one neuron.
If images were stored in RGB format, each pixel would be represented by three neurons, one for each colour channel.
Increasing the number of neurons in the network, theoretically should improve the memory capacity of the network.
However, it is very likely this would only work if the digits were different colours.
As the 0, 6 and 8 are the closest digits, it is likely an improvement could occur if the 0 was all red, the 6 all blue, and the 8 all green.
However, this would mean the input would have to be the same colour, otherwise it is unlikely the network would find the correct minima.

## Further Reading
Inspired by https://towardsdatascience.com/hopfield-networks-neural-memory-machines-4c94be821073

Research: 
https://www.youtube.com/watch?v=Rs1XMS8NqB4&ab_channel=ArtificialIntelligence-AllinOne

Storkey learning rule: 
https://www.reddit.com/r/pythontips/comments/yw97o9/weight_matrix_according_to_storkey_learning_rule/
https://stats.stackexchange.com/questions/276889/whats-wrong-with-my-algorithm-for-implementing-the-storkey-learning-rule-for-ho
https://github.com/drussellmrichie/hopfield_network/blob/master/Hopfield%20Network%20of%20memes--Storkey%20Learning%20Rule.ipynb
https://link.springer.com/chapter/10.1007/bfb0020196
 
 
