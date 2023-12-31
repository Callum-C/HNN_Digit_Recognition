# HNN_Digit_Recognition
Utilising Hopfield Neural Networks to recognise digits 0 - 9.

A network is initialised and trained on a subset of the 10 available digits.

The network excels at recognising digits it has learned so long as the number it has "memorised" is no more than 2.

The network is capable of recalling one of two memories even if the pattern is severely noisy or corrupted.

![Recall-4-425](https://github.com/Callum-C/HNN_Digit_Recognition/assets/60474698/92987b7e-5cdf-494d-a46a-ae7f51f1e7f4)  ![Recall-7-425](https://github.com/Callum-C/HNN_Digit_Recognition/assets/60474698/91ce3d4e-8260-46f5-b2a0-00f781d68176)

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

## Results
For the below test, a Hopfield Neural Network was initialised and trained on the ideal group of digits as determined by the Hamming distance.
This group being the 0, the 4 and the 7. Salt and pepper noise was randomly added to each source image in turn and set as the network's starting state.
While the amount of noise remained constant for all 6 images, the cells that were flipped were chosen at random. It is possible, although unlikely, for one cell to have been flipped multiple times.
As shown below, the Storkey training method has much greater success restoring the 0 and the 7 digit, but still falls into the same spurious minima while attempting to restore / recall the 4.
While the 0 falls into something of a spurious minima, it is still recognisable as a 0. On the other hand the Hebbian method fails to successfully recall any of the 3 digits to a recognisable state.

### Spurious Minima - Hebbian Training Method
![Spurious-Hebbian-0](https://github.com/Callum-C/HNN_Digit_Recognition/assets/60474698/45ede62d-20ab-443e-b8ed-aea611ebe709)  ![Spurious-Hebbian-4](https://github.com/Callum-C/HNN_Digit_Recognition/assets/60474698/020172f2-8252-4a07-b8bb-a289970b58b9)  ![Spurious-Hebbian-7](https://github.com/Callum-C/HNN_Digit_Recognition/assets/60474698/fd6b7c9d-1d59-4ba8-8297-3a9fdf55c289)


### Spurious Minima - Storkey Training Method
![Spurious-Storkey-0](https://github.com/Callum-C/HNN_Digit_Recognition/assets/60474698/6d68355c-2e56-4075-b605-b72e3a673188)  ![Spurious-Storkey-4](https://github.com/Callum-C/HNN_Digit_Recognition/assets/60474698/35aab213-bd18-48d5-a6ff-db4ffbcb76b0)  ![Spurious-Storkey-7](https://github.com/Callum-C/HNN_Digit_Recognition/assets/60474698/601fb250-804b-4d0a-a67d-9672a68c2887)

## Summary
Hopfield Neural Networks are a great example of assosciative memory, capable of restoring a full memory from partial, in-complete or corrupted data. 
While likely to excel in domains where memories are disctinctively different, it has been shown they struggle significantly in applications where the memories are so similar.
With an advertised capacity of 0.15 * N, at 2500 neurons and failing to recall 3 or more digits, the HNN falls short of expectations.
The Storkey Training Method is a definitive improvement over the Hebbian method, but still leaves something to be desired.

## Further Reading
Inspired by https://towardsdatascience.com/hopfield-networks-neural-memory-machines-4c94be821073

Research: 
https://www.youtube.com/watch?v=Rs1XMS8NqB4&ab_channel=ArtificialIntelligence-AllinOne

Storkey learning rule: 
https://www.reddit.com/r/pythontips/comments/yw97o9/weight_matrix_according_to_storkey_learning_rule/
https://stats.stackexchange.com/questions/276889/whats-wrong-with-my-algorithm-for-implementing-the-storkey-learning-rule-for-ho
https://github.com/drussellmrichie/hopfield_network/blob/master/Hopfield%20Network%20of%20memes--Storkey%20Learning%20Rule.ipynb
https://link.springer.com/chapter/10.1007/bfb0020196
 
 
