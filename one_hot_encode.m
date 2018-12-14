% EECS545 Final Project 32
% Haochen Wu, Yen-Yu Hsu, Shichao Zeng
% Online Sequential Extreme Learning Machine on Handwritten Digits Classification
% One Hot Encoder
function labels_oh = one_hot_encode(labels, num_classes)

labels_oh=-1*ones(size(labels,1),num_classes);
for i = 0:num_classes-1
    row=(labels==i);
    labels_oh(row,i+1) = 1;
end
assert(isequal(size(labels_oh), [size(labels, 1),i+1]))

end
