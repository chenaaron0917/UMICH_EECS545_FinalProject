% EECS545 Final Project 32
% Haochen Wu, Yen-Yu Hsu, Shichao Zeng
% Online Sequential Extreme Learning Machine on Handwritten Digits Classification
% Plot misclassification rate with different Hidden Neurons(collected data)
clc
clear
close all
figure(1)
load 'E512_64.mat'
plot(E)
hold on
load 'E1024_1536_64.mat'
plot(E)
hold on
load 'E2048_64.mat'
plot(E)
hold on
load 'E4096_64.mat'
plot(E)
xlabel('Number of Batches')
ylabel('Misclassification Rate')
legend('512 Hidden Neurons','1024 Hidden Neurons','2048 Hidden Neurons','4096 Hidden Neurons')
