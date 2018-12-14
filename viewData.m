% EECS545 Final Project 32
% Haochen Wu, Yen-Yu Hsu, Shichao Zeng
% Online Sequential Extreme Learning Machine on Handwritten Digits Classification
% Randomly select each digit to visualize
clc;clear;close all
[trainimages, trainlabels, dim] = mnist_parse('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');
[testimages, testlabels,dim] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');
index=[];
for i=0:9
    ind=find(trainlabels==i);
    k=randi(size(ind,1));
    index=[index ind(k)];
end
figure(1)
for j=1:10
    subplot(2,5,j)
    i=index(j);
    imagesc(reshape(trainimages(:,i),[28 28]))
    title(num2str(trainlabels(i)))
end