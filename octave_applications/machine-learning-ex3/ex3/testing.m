load('ex3weights.mat');

img4 = reshape(mat2gray(imread("four.jpg")), [1,400]);
pred4 = predict(Theta1, Theta2, img4)

img9 = reshape(mat2gray(imread("nine.jpg")), [1,400]);
pred9 = predict(Theta1, Theta2, img9)


img3 = reshape(mat2gray(imread("three.jpg")), [1,400]);
pred3 = predict(Theta1, Theta2, img3)

% Not a very good predictor..
